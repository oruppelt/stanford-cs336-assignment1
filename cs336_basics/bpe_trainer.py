# bpe_trainer.py
# added delta incremental updates for pair frequencies using chatgpt.

# bpe_trainer.py
from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Tuple, Dict, BinaryIO
from collections import defaultdict

import regex as re  # supports \p{L}, \p{N}, etc.

# ---------- Shared pattern ----------
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
def build_token_pattern(special_tokens: list[str]) -> re.Pattern:
    # Escape specials and match them first (atomic)
    specials_alt = "|".join(re.escape(s) for s in special_tokens) if special_tokens else ""
    base_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if specials_alt:
        return re.compile(f"(?:{specials_alt})|{base_pat}")
    return re.compile(base_pat)

class BPETrainer:
    """
    Owns corpus scanning, pre-tokenization, stats, and merge learning.
    Trains byte-level BPE and can export (vocab, merges).
    Reference implementation: full recount each step with deterministic tie-break.
    """

    def __init__(self, byte_level: bool = True):
        self.PAT = PAT
        self.special_tokens: List[str] = ["<|endoftext|>"]
        self.byte_level = byte_level

        # Training state
        self.word_counts: Counter[bytes] = Counter()           # token (bytes) -> count
        self.segmented_words: Dict[bytes, List[bytes]] = {}    # token -> list of byte symbols
        self.vocab_index: Dict[bytes, int] = {}                # symbol -> stable id (non-special)
        self.merges: List[Tuple[bytes, bytes]] = []

        # Pair counts (rebuilt every iteration)
        self.pair_counts: Counter[Tuple[bytes, bytes]] = Counter()

        self.pair_freq: Counter[tuple[bytes, bytes]] = Counter()      # live pair frequencies
        self.pair_occ: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)  # which words contain a pair

    # ---------- Public pipeline (serial pretokenization) ----------

    def pretokenize(self, corpus: Iterable[str]) -> "BPETrainer":
        """
        Serial pretokenization: build word_counts (bytes -> freq).
        Special-token strings are removed from text so they don't affect merges.
        """
        wc: Counter[bytes] = Counter()
        pat = build_token_pattern(self.special_tokens)
        specials_s = set(self.special_tokens)

        for line in corpus:
            line = line.rstrip("\n")
            # Remove any occurrences of special-token strings
            for s in specials_s:
                line = line.replace(s, "")

            for m in pat.finditer(line):
                tok = m.group(0)
                if not tok:
                    continue
                wc[tok.encode("utf-8")] += 1

        # init state
        self.word_counts = wc
        self.segmented_words = {
            w: ([w] if w.decode("utf-8") in specials_s else [bytes([b]) for b in w])
            for w in wc
        }
        self.vocab_index = {bytes([b]): b for b in range(256)}  # base bytes
        return self

    def compute_pair_stats(self) -> "BPETrainer":
        """One-time build of pair frequencies and occurrences."""
        self.pair_freq.clear()
        self.pair_occ.clear()
        for w, freq in self.word_counts.items():
            seq = self.segmented_words[w]
            for a, b in zip(seq, seq[1:]):
                p = (a, b)
                self.pair_freq[p] += freq
                self.pair_occ[p].add(w)
        return self

    def fit_to_vocab_size(self, vocab_size: int, special_tokens: list[str], progress: bool = True) -> "BPETrainer":
        target_non_special = max(0, vocab_size - len(special_tokens))

        # build initial pair stats (once)
        if not self.pair_freq:
            self.compute_pair_stats()

        while len(self.vocab_index) < target_non_special and self.pair_freq:
            # choose best by frequency, tie-break by lexicographically GREATER pair
            maxf = max(self.pair_freq.values())
            # NOTE: max(...) over tuples implements lexicographically greater
            best_pair = max((p for p, f in self.pair_freq.items() if f == maxf))

            # apply merge with local delta updates
            self._apply_merge_delta_simple(best_pair)
            self.merges.append(best_pair)

            if progress and (len(self.merges) % 1000 == 0):
                pass  # add logging if you want

        return self

    def export_vocab_and_merges(
        self, special_tokens: List[str], vocab_size: int
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Build final vocab (IDs -> bytes) with special tokens first, then symbols by stable index.
        Caps size at vocab_size.
        """
        vocab: Dict[int, bytes] = {i: s.encode("utf-8") for i, s in enumerate(special_tokens)}
        offset = len(special_tokens)

        for sym, idx in sorted(self.vocab_index.items(), key=lambda kv: kv[1]):
            tid = offset + idx
            if tid >= vocab_size:
                break
            vocab[tid] = sym

        return vocab, list(self.merges)

    # ---------- Private helpers ----------

    def _init_state_from_counts(self, wc: Counter[bytes]) -> None:
        """Initialize trainer state from precomputed word_counts (bytes -> freq)."""
        self.word_counts = wc
        # Token -> sequence of single-byte symbols
        self.segmented_words = {w: [bytes([b]) for b in w] for w in wc}
        # Base 256-byte vocab (ids 0..255 by byte value)
        self.vocab_index = {bytes([b]): b for b in range(256)}

    def _apply_merge_full(self, pair: Tuple[bytes, bytes]) -> None:
        """Apply merge (a,b)->a+b to all words (no delta structures)."""
        a, b = pair
        merged = a + b
        if merged not in self.vocab_index:
            self.vocab_index[merged] = len(self.vocab_index)

        for w, seq in self.segmented_words.items():
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

    def _apply_merge_delta_simple(self, pair: tuple[bytes, bytes]) -> None:
        """
        Merge (a,b)->a+b by updating only words that contain the pair.
        Recompute pairs for each affected word and update pair_freq/pair_occ.
        """
        a, b = pair
        merged = a + b
        if merged not in self.vocab_index:
            self.vocab_index[merged] = len(self.vocab_index)

        affected_words = list(self.pair_occ.get(pair, set()))
        # after we process them, this pair’s occurrences disappear
        # self.pair_occ[pair].clear()
        # self.pair_freq[pair] = 0  

        for w in affected_words:
            seq = self.segmented_words[w]
            if not seq:
                continue

            # ----- old pairs for this word (before) -----
            old_pairs = list(zip(seq, seq[1:]))

            # ----- apply merge locally -----
            i = 0
            new_seq: list[bytes] = []
            changed = False
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                    changed = True
                else:
                    new_seq.append(seq[i])
                    i += 1

            if not changed:
                # stale membership; skip
                continue

            # ----- new pairs for this word (after) -----
            new_pairs = list(zip(new_seq, new_seq[1:]))
            freq_w = self.word_counts[w]

            # ----- delta update: remove old contributions -----
            for p in old_pairs:
                self.pair_freq[p] -= freq_w
                if self.pair_freq[p] <= 0:
                    # fully remove to keep dict small; also clear occ set
                    self.pair_freq.pop(p, None)
                    s = self.pair_occ.get(p)
                    if s is not None:
                        s.discard(w)
                        if not s:
                            self.pair_occ.pop(p, None)
                else:
                    # still present globally; ensure w no longer listed for p
                    s = self.pair_occ.get(p)
                    if s is not None:
                        s.discard(w)
                        if not s:
                            self.pair_occ.pop(p, None)

            # ----- delta update: add new contributions -----
            for p in new_pairs:
                self.pair_freq[p] += freq_w
                self.pair_occ.setdefault(p, set()).add(w)

            # save updated segmentation
            self.segmented_words[w] = new_seq

        self.pair_occ.pop(pair, None)
        if self.pair_freq.get(pair, 0) <= 0:
            self.pair_freq.pop(pair, None)
    
# -------- Parallel pretokenization helpers (optional) --------

def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts ending at a delimiter; may return fewer chunks.
    """
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 1 << 20  # 1MB probe for fewer syscalls
    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini_chunk_size)
            if buf == b"":
                chunk_boundaries[bi] = file_size
                break
            j = buf.find(split_special_token)
            if j != -1:
                chunk_boundaries[bi] = pos + j
                break
            pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _count_slice_from_file(path: str, start: int, end: int, specials_s: set[str]) -> Counter[str]:
    """
    Worker: count tokens (string keys) in one slice. Strings are cheaper; convert once later.
    """
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # strip specials so their fragments never appear
    for s in specials_s:
        chunk = chunk.replace(s, " ")

    c = Counter()
    for m in PAT.finditer(chunk):
        tok = m.group(0)
        if tok:
            c[tok] += 1
    return c


def parallel_counts_from_boundaries(
    input_path: str, boundaries: list[int], special_tokens: List[str], max_workers: int
) -> Counter[bytes]:
    """
    Map→Reduce: produce global word_counts (bytes -> freq) from chunk boundaries.
    """
    totals_str = Counter()
    specials_s = set(special_tokens)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_count_slice_from_file, input_path, s, e, specials_s)
            for s, e in zip(boundaries[:-1], boundaries[1:])
        ]
        for fu in futs:
            totals_str.update(fu.result())

    # Convert keys to bytes once (cheaper than encoding per token)
    return Counter({k.encode("utf-8"): v for k, v in totals_str.items()})


# -------- Public training function (deliverable) --------

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    *,
    num_processes: int | None = None,
    delimiter: bytes = b"<|endoftext|>",
    parallel: bool = False,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE tokenizer and returns (vocab_id_to_bytes, merges).

    - Ensures base 256-byte alphabet + specials are included within vocab_size.
    - If parallel=True, uses chunked pretokenization; otherwise serial.
    """
    required_min = len(special_tokens) + 256
    if vocab_size < required_min:
        raise ValueError(
            f"vocab_size must be at least {required_min} "
            f"(= {len(special_tokens)} specials + 256 base bytes)"
        )

    trainer = BPETrainer(byte_level=True)
    trainer.special_tokens = list(special_tokens)

    if parallel:
        if num_processes is None:
            num_processes = os.cpu_count() or 4
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, delimiter)
        wc = parallel_counts_from_boundaries(input_path, boundaries, special_tokens, num_processes)
        trainer._init_state_from_counts(wc)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            trainer.pretokenize(f)

    trainer.fit_to_vocab_size(vocab_size, special_tokens)
    return trainer.export_vocab_and_merges(special_tokens, vocab_size)


# ---- Example main (optional) ----
if __name__ == "__main__":
    vocab, merges = train_bpe_tokenizer(
        input_path="../data/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        parallel=False,
        num_processes=4,
    )
    print(f"Returned vocab size: {len(vocab)} | Merges learned: {len(merges)}")

    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        if b"<|" in word_bytes:
            print(f"Warning: Found special token in word: {word_bytes}")