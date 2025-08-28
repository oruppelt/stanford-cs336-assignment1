from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import regex as re  # pip install regex

# GPT-2-style pretokenizer pattern (from tiktoken PR)
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _specials_union_pattern(special_tokens: List[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    escaped = [re.escape(s) for s in special_tokens]
    return re.compile("|".join(escaped))


class BPETrainer:
    """
    Reference byte-level BPE trainer (simple & deterministic):
      - Pre-tokenizes with GPT-2 regex *within* segments split on special tokens.
      - Never lets specials influence training (they are excluded from counts).
      - Full recount every iteration; tie-break by lexicographically GREATER pair.
    Produces (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]).
    """

    def __init__(self, *, special_tokens: List[str] | None = None) -> None:
        self.special_tokens: List[str] = list(special_tokens or [])
        self.word_counts: Counter[bytes] = Counter()           # pretoken bytes -> freq
        self.segmented_words: Dict[bytes, List[bytes]] = {}     # token bytes -> list of byte symbols
        self.vocab_index: Dict[bytes, int] = {}                 # symbol -> stable id (0..)
        self.merges: List[Tuple[bytes, bytes]] = []

    # -------- Pretokenization --------

    def pretokenize_from_string(self, text: str) -> "BPETrainer":
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

        self._init_state_from_counts(wc)
        return self

    def pretokenize(self, lines: Iterable[str]) -> "BPETrainer":
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

    # -------- Merge training (full recount, deterministic) --------

    def _recount_pairs(self) -> Counter[Tuple[bytes, bytes]]:
        counts: Counter[Tuple[bytes, bytes]] = Counter()
        for w, freq in self.word_counts.items():
            seq = self.segmented_words[w]
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += freq
        return counts

    def _apply_merge_full(self, pair: Tuple[bytes, bytes]) -> None:
        a, b = pair
        merged = a + b
        if merged not in self.vocab_index:
            self.vocab_index[merged] = len(self.vocab_index)
        for w, seq in list(self.segmented_words.items()):
            i = 0
            out: List[bytes] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            self.segmented_words[w] = out

    def fit_to_vocab_size(self, vocab_size: int, progress: bool = False) -> "BPETrainer":
        """Greedy BPE: full recount each step; tie-break by lexicographically GREATER pair."""
        target_non_special = max(0, vocab_size - len(self.special_tokens))
        while len(self.vocab_index) < target_non_special:
            pair_counts = self._recount_pairs()
            if not pair_counts:
                break
            maxf = max(pair_counts.values())
            best_pair = max(p for p, f in pair_counts.items() if f == maxf)
            self._apply_merge_full(best_pair)
            self.merges.append(best_pair)
            if progress and len(self.merges) % 1000 == 0:
                pass
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


# -------- Public training function (deliverable) --------

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return (vocab, merges).

    This reference implementation follows the assignment:
      - Initial vocab = 256 bytes.
      - Special tokens are added to the vocabulary but do not affect merges.
      - Pretokenization splits on specials; no merges across those boundaries.
      - Tie-break rule: lexicographically GREATER pair among max-frequency ties.
    """
    required_min = 256 + len(special_tokens)
    if vocab_size < required_min:
        raise ValueError(
            f"vocab_size must be at least {required_min} (= 256 bytes + {len(special_tokens)} specials)"
        )

    trainer = BPETrainer(special_tokens=special_tokens)
    with open(input_path, "r", encoding="utf-8") as f:
        trainer.pretokenize(f)
    trainer.fit_to_vocab_size(vocab_size)
    return trainer.export_vocab_and_merges(vocab_size)


# ---- Optional CLI ----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--vocab_size", type=int, default=2000)
    ap.add_argument("--special", action="append", default=["<|endoftext|>"])
    args = ap.parse_args()

    vocab, merges = train_bpe_tokenizer(args.input_path, args.vocab_size, args.special)
    print(f"Vocab size: {len(vocab)}  |  Merges learned: {len(merges)}")
