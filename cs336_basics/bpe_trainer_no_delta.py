# bpe_trainer.py
# This needs to be rewritten to paralellize the pre-tokenization process.


from collections import Counter
from typing import Iterable, List, Tuple, Dict
import json
import regex as re  # supports \p{L}, \p{N}, etc.
from scalene import scalene_profiler

class BPETrainer:
    """
    Owns corpus scanning, pre-tokenization, stats, and merge learning.
    Trains byte-level BPE and can export (vocab, merges).
    """

    def __init__(self, byte_level: bool = True):
        # GPT-style pattern; works with 'regex' (not 're')
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_tokens: List[str] = ['<|endoftext|>']  # will be overwritten by wrapper
        self.byte_level = byte_level

        # Training state
        self.word_counts: Counter[bytes] = Counter()        # token (bytes) -> count
        self.segmented_words: Dict[bytes, List[bytes]] = {}  # token (bytes) -> list of byte-symbols
        self.pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab_index: Dict[bytes, int] = {}             # symbol -> stable id (non-special)

        # Dirty flag
        self._pairs_dirty = True

    # ---------- Public pipeline ----------

    def pretokenize(self, corpus: Iterable[str]) -> "BPETrainer":
        """
        Scans the corpus once and stores token frequency stats.
        Special tokens are ignored for training (they don't affect merges).
        """
        wc: Counter[bytes] = Counter()
        specials = set(self.special_tokens)

        i = 0
        for line in corpus:
            line = line.rstrip("\n")
            # skip lines that are exactly a special token
            if line in specials:
                continue
            for m in re.finditer(self.PAT, line):
                tok = m.group(0)
                if not tok or tok in specials:
                    continue
                tok_bytes = tok.encode("utf-8")
                wc[tok_bytes] += 1
            i += 1
            if i % 100_000 == 0:
                print(f"Processed {i} lines; current unique count: {len(wc)}")

        self.word_counts = wc

        # Initialize segmentation: each token -> list of single-byte symbols
        self.segmented_words = {
            w: [bytes([b]) for b in w]  # w is already bytes; iterating yields ints
            for w in wc
        }

        # Initial vocab = all unique single-byte symbols observed
        base_symbols = [bytes([b]) for b in range(256)]
        self.vocab_index = {sym: i for i, sym in enumerate(base_symbols)}

        self._pairs_dirty = True
        return self

    def compute_pair_stats(self) -> "BPETrainer":
        """Rebuild pair frequency counts across current segmentations."""
        self.pair_counts = self._compute_pair_counts()
        self._pairs_dirty = False
        return self

    def fit_to_vocab_size(self, vocab_size: int, special_tokens: List[str], progress: bool = True) -> "BPETrainer":
        """
        Greedy BPE loop that stops when len(non-special symbols) + len(special_tokens) reaches vocab_size.
        """
        target_non_special = max(0, vocab_size - len(special_tokens))
        while len(self.vocab_index) < target_non_special:
            if self._pairs_dirty:
                self.compute_pair_stats()
            if not self.pair_counts:  # nothing left to merge
                break
            best_pair, _ = self.pair_counts.most_common(1)[0]
            self._apply_merge(best_pair)
            self.merges.append(best_pair)
            self._pairs_dirty = True
            if progress and (len(self.merges) % 1000 == 0):
                print(f"Merged 1000 pairs; current vocab size: {len(self.vocab_index)}")
        return self

    def export_vocab_and_merges(
        self,
        special_tokens: List[str],
        vocab_size: int
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Build final vocab (IDs -> bytes) with special tokens first, then symbols by stable index.
        Caps size at vocab_size.
        """
        vocab: Dict[int, bytes] = {i: s.encode("utf-8") for i, s in enumerate(special_tokens)}
        offset = len(special_tokens)

        # Non-special symbols in stable order
        items = sorted(self.vocab_index.items(), key=lambda kv: kv[1])
        for sym, idx in items:
            tok_id = offset + idx
            if tok_id >= vocab_size:
                break
            vocab[tok_id] = sym

        return vocab, list(self.merges)

    # ---------- Private helpers ----------

    def _compute_pair_counts(self) -> Counter:
        counts: Counter[Tuple[bytes, bytes]] = Counter()
        for w, freq in self.word_counts.items():
            seq = self.segmented_words[w]
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += freq
        return counts

    def _apply_merge(self, pair: Tuple[bytes, bytes]) -> None:
        """
        Fuse occurrences of 'pair' into a single symbol across all segmentations.
        Update vocab_index with the new merged symbol if needed.
        """
        merged_symbol = pair[0] + pair[1]
        if merged_symbol not in self.vocab_index:
            self.vocab_index[merged_symbol] = len(self.vocab_index)

        for w, seq in self.segmented_words.items():
            i = 0
            out: List[bytes] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    out.append(merged_symbol)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            self.segmented_words[w] = out

        self._pairs_dirty = True

    # ---------- Optional utilities ----------

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "word_counts": {k.decode("latin1"): v for k, v in self.word_counts.items()},
            "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in self.merges],
            "special_tokens": list(self.special_tokens),
            "byte_level": self.byte_level,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load_checkpoint(path: str) -> "BPETrainer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        t = BPETrainer(byte_level=obj["byte_level"])
        t.special_tokens = obj["special_tokens"]
        t.word_counts = Counter({k.encode("latin1"): v for k, v in obj["word_counts"].items()})
        t.merges = [(a.encode("latin1"), b.encode("latin1")) for a, b in obj["merges"]]
        # segmented_words and pair_counts would need rebuild from word_counts and merges if desired
        return t


# -------- Public training function (deliverable) --------

def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: List[str]):
    required_min = len(special_tokens) + 256
    if vocab_size < required_min:
        raise ValueError(
            f"vocab_size must be at least {required_min} "
            f"(= {len(special_tokens)} specials + 256 base bytes)."
        )

    trainer = BPETrainer(byte_level=True)
    trainer.special_tokens = list(special_tokens)  # ensures pretokenize ignores them

    with open(input_path, "r", encoding="utf-8") as f:
        trainer.pretokenize(f).compute_pair_stats().fit_to_vocab_size(vocab_size, special_tokens)

    vocab, merges = trainer.export_vocab_and_merges(special_tokens, vocab_size)
    return vocab, merges


# ---- Example main (optional) ----
if __name__ == "__main__":
    scalene_profiler.start()
    vocab, merges = train_bpe_tokenizer(
        input_path="../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=2_000,
        special_tokens=["<|endoftext|>"],
    )
    scalene_profiler.stop()

    # Print some stats
    print(f"Vocab size (requested): {50_000} | Special tokens: {len(vocab) - 256}")
    print(f"Vocab size (returned): {len(vocab)} | Merges learned: {len(merges)}")
