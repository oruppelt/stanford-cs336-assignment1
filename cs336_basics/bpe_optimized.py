from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Set

import regex as re  # pip install regex

import psutil
import gc

# GPT-2-style pretokenizer pattern (from tiktoken PR)
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def check_memory():
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    if memory_mb > 8000:  # 8GB limit
        print(f"Memory usage: {memory_mb:.1f} MB")
        gc.collect()


def _specials_union_pattern(special_tokens: List[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    escaped = [re.escape(s) for s in special_tokens]
    return re.compile("|".join(escaped))


class OptimizedBPETrainer:
    """
    Optimized byte-level BPE trainer with incremental pair counting:
      - Pre-tokenizes with GPT-2 regex *within* segments split on special tokens.
      - Never lets specials influence training (they are excluded from counts).
      - Incremental pair counting for significant speedup.
      - Tie-break by lexicographically GREATER pair.
    Produces (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]).
    """

    def __init__(self, *, special_tokens: List[str] | None = None) -> None:
        self.special_tokens: List[str] = list(special_tokens or [])
        self.word_counts: Counter[bytes] = Counter()           # pretoken bytes -> freq
        self.segmented_words: Dict[bytes, List[bytes]] = {}     # token bytes -> list of byte symbols
        self.vocab_index: Dict[bytes, int] = {}                 # symbol -> stable id (0..)
        self.merges: List[Tuple[bytes, bytes]] = []

        # Optimization: cached pair counts and reverse indices
        self.pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
        self.pair_locations: Dict[Tuple[bytes, bytes], Set[bytes]] = defaultdict(set)  # pair -> words containing it
        self._pairs_dirty = True  # flag to track if we need to recompute from scratch

    # -------- Pretokenization --------

    def pretokenize_from_string(self, text: str) -> "OptimizedBPETrainer":
        """Split *on* special tokens, then GPT-2-pretokenize each segment.
        Special tokens do not contribute to counts or merges.
        """
        splitter = _specials_union_pattern(self.special_tokens)
        segments = re.split(splitter, text) if splitter else [text]

        wc: Counter[bytes] = Counter()
        for seg in segments:
            if not seg:
                continue
            for m in PAT.finditer(seg):
                tok = m.group(0)
                if tok:
                    wc[tok.encode("utf-8")] += 1
                    # check_memory()

        self._init_state_from_counts(wc)
        return self

    def pretokenize(self, lines: Iterable[str]) -> "OptimizedBPETrainer":
        """Streaming version: join lines and delegate.
        The datasets in tests are small; for very large corpora prefer the chunked parallel helper.
        """
        text = "".join(lines)
        return self.pretokenize_from_string(text)

    def _init_state_from_counts(self, wc: Counter[bytes]) -> None:
        self.word_counts = wc
        # Each pretoken becomes a sequence of *single bytes* symbols.
        self.segmented_words = {w: [bytes([b]) for b in w] for w in wc}
        # Base 256-byte alphabet, ids 0..255 by byte value.
        self.vocab_index = {bytes([b]): b for b in range(256)}
        self._pairs_dirty = True

    # -------- Optimized merge training with incremental counting --------

    def _compute_initial_pairs(self) -> None:
        """Compute all pair counts from scratch and build reverse index."""
        self.pair_counts.clear()
        self.pair_locations.clear()

        for word, freq in self.word_counts.items():
            seq = self.segmented_words[word]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                self.pair_counts[pair] += freq
                self.pair_locations[pair].add(word)

        self._pairs_dirty = False

    def _get_pairs_in_word(self, word_seq: List[bytes]) -> List[Tuple[bytes, bytes]]:
        """Get all adjacent pairs in a word sequence."""
        return [(word_seq[i], word_seq[i + 1]) for i in range(len(word_seq) - 1)]

    def _apply_merge_incremental(self, pair: Tuple[bytes, bytes]) -> None:
        """Apply merge with incremental pair count updates."""
        a, b = pair
        merged = a + b

        # Add merged symbol to vocabulary
        if merged not in self.vocab_index:
            self.vocab_index[merged] = len(self.vocab_index)

        # Get all words that contain this pair
        affected_words = list(self.pair_locations[pair])

        # Update each affected word
        for word in affected_words:
            seq = self.segmented_words[word]
            freq = self.word_counts[word]

            # Get pairs before merge
            old_pairs = self._get_pairs_in_word(seq)

            # Apply merge to sequence
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            # Get pairs after merge
            new_pairs = self._get_pairs_in_word(new_seq)

            # Update pair counts and locations
            # Remove old pairs
            for old_pair in old_pairs:
                self.pair_counts[old_pair] -= freq
                if self.pair_counts[old_pair] <= 0:
                    del self.pair_counts[old_pair]
                self.pair_locations[old_pair].discard(word)
                if not self.pair_locations[old_pair]:
                    del self.pair_locations[old_pair]

            # Add new pairs
            for new_pair in new_pairs:
                self.pair_counts[new_pair] += freq
                self.pair_locations[new_pair].add(word)

            # Update word segmentation
            self.segmented_words[word] = new_seq

    def fit_to_vocab_size(self, vocab_size: int, progress: bool = False) -> "OptimizedBPETrainer":
        """Greedy BPE with incremental pair counting; tie-break by lexicographically GREATER pair."""
        target_non_special = max(0, vocab_size - len(self.special_tokens))

        # Initialize pair counts if needed
        if self._pairs_dirty:
            self._compute_initial_pairs()

        merge_count = 0
        while len(self.vocab_index) < target_non_special:
            if not self.pair_counts:
                break

            # Find most frequent pair(s)
            max_freq = max(self.pair_counts.values())
            best_pair = max(pair for pair, freq in self.pair_counts.items() if freq == max_freq)

            # Apply merge incrementally
            self._apply_merge_incremental(best_pair)
            self.merges.append(best_pair)
            merge_count += 1
            check_memory()
            if progress and merge_count % 1000 == 0:
                print(f"Completed {merge_count} merges, vocab size: {len(self.vocab_index)}")

        return self

    # -------- Export --------

    def export_vocab_and_merges(self, vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """Return (vocab_id_to_bytes, merges) with specials first, then symbols by stable id."""
        vocab: Dict[int, bytes] = {i: s.encode("utf-8") for i, s in enumerate(self.special_tokens)}
        offset = len(self.special_tokens)
        for sym, idx in sorted(self.vocab_index.items(), key=lambda kv: kv[1]):
            tok_id = offset + idx
            if tok_id >= vocab_size:
                break
            vocab[tok_id] = sym
        return vocab, list(self.merges)


# -------- Updated public training function --------

def train_bpe_tokenizer_optimized(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train an optimized byte-level BPE tokenizer and return (vocab, merges).

    This optimized implementation uses incremental pair counting for significant speedup:
      - Initial vocab = 256 bytes.
      - Special tokens are added to the vocabulary but do not affect merges.
      - Pretokenization splits on specials; no merges across those boundaries.
      - Tie-break rule: lexicographically GREATER pair among max-frequency ties.
      - Incremental pair counting instead of full recount each iteration.
    """
    required_min = 256 + len(special_tokens)
    if vocab_size < required_min:
        raise ValueError(
            f"vocab_size must be at least {required_min} (= 256 bytes + {len(special_tokens)} specials)"
        )

    trainer = OptimizedBPETrainer(special_tokens=special_tokens)
    with open(input_path, "r", encoding="utf-8") as f:
        trainer.pretokenize(f)
    trainer.fit_to_vocab_size(vocab_size, progress=True)
    return trainer.export_vocab_and_merges(vocab_size)


# ---- Optional CLI ----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--vocab_size", type=int, default=2000)
    ap.add_argument("--special", action="append", default=["<|endoftext|>"])
    args = ap.parse_args()

    vocab, merges = train_bpe_tokenizer_optimized(args.input_path, args.vocab_size, args.special)
    print(f"Vocab size: {len(vocab)}  |  Merges learned: {len(merges)}")
