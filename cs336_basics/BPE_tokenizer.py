### This is the first version before actually carefully reading the instructions.
# BPE Tokenizer Implementation
# Different version will be in `BPE_trainer.py`


from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import regex as re
import json
from collections import Counter, defaultdict

# ----------------------------
# Runtime Tokenizer (frozen)
# ----------------------------
@dataclass(frozen=True)
class BPETokenizer:
    merges: Tuple[Tuple[bytes, bytes], ...]              
    vocab: Tuple[bytes, ...]                             
    special_tokens: Tuple[str, ...] = ()                 
    byte_level: bool = False                              # byte-level BPE by default

    def encode(self, text: str) -> List[int]:
        """Apply merges to text and map to vocab IDs. No training state used."""
        # TODO: implement: normalize -> split -> bytes -> iterative merges -> ids
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Map ids back to byte sequences and join."""
        # TODO: implement reverse mapping: ids -> bytes -> text
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        """Serialize tokenizer for saving to disk."""
        payload = {
            "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in self.merges],
            "vocab": [tok.decode("latin1") for tok in self.vocab],
            "special_tokens": list(self.special_tokens),
            "byte_level": self.byte_level,
        }
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def from_bytes(blob: bytes) -> "BPETokenizer":
        obj = json.loads(blob.decode("utf-8"))
        merges = tuple((a.encode("latin1"), b.encode("latin1")) for a, b in obj["merges"])
        vocab = tuple(s.encode("latin1") for s in obj["vocab"])
        return BPETokenizer(
            merges=merges,
            vocab=vocab,
            special_tokens=tuple(obj["special_tokens"]),
            byte_level=bool(obj["byte_level"]),
        )


# ----------------------------
# Trainer (stateful)
# ----------------------------
class BPETrainer:
    """
    Owns corpus scanning, pre-tokenization, stats, and merge learning.
    Produces a BPETokenizer at the end.
    """

    def __init__(
        self,
        # token_pattern: Optional[str] = r"\S+",
        # special_tokens: Optional[Iterable[str]] = None,
        byte_level: bool = True,
        # lowercase: bool = False,
    ):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_tokens = ['<|endoftext|>']
        self.byte_level = byte_level
        # self.lowercase = lowercase

        # Training state
        self.word_counts: Counter[str] = Counter()          # pretokenized words -> counts
        self.segmented_words: Dict[str, List[bytes]] = {}   # word -> current segmentation (list of byte “symbols”)
        self.pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab_index: Dict[bytes, int] = {}             # symbol -> id

        # Dirty flags
        self._pairs_dirty = True

    # ---------- Public pipeline ----------

    def pretokenize(self, corpus: Iterable[str]) -> "BPETrainer":
        """
        Scans the corpus once and stores word frequency stats.
        This is where your regex splitting lives.
        """
        bytestring = self.byte_level

        wc = Counter()
        for i in range(len(corpus)):
            if corpus[i].rstrip() in self.special_tokens:
                token_bytes = corpus[i].rstrip().encode('utf-8')
                # if bytestring:
                key = token_bytes
                # else:
                    # key = tuple([self.vocab.get(token_bytes)])  
                    # pass
                wc[key] = wc.get(key, 0) + 1
            else:
                for match in re.finditer(self.PAT, corpus[i]):
                    token = match.group()
                    if token:
                        token_bytes = token.encode('utf-8')
                        if corpus:
                            key = token_bytes
                        else:
                            key = tuple(token_bytes)
                        wc[key] = wc.get(key, 0) + 1

        self.word_counts = wc
        # Initialize each word as sequence of bytes (byte-level) or chars (unicode-level)
        self.segmented_words = {
            w: (list(w) if self.byte_level else list(w))  # placeholder; treat as bytes
            for w in wc
        }
        # Convert ints -> 1-byte bytes for consistency
        self.segmented_words = {w: [bytes([b]) for b in seq] for w, seq in self.segmented_words.items()}

        # Initialize vocab with all single bytes encountered (+ special tokens later)
        symbols = set(sym for seq in self.segmented_words.values() for sym in seq)
        self.vocab_index = {sym: i for i, sym in enumerate(sorted(symbols))}
        self._pairs_dirty = True
        return self

    def compute_pair_stats(self) -> "BPETrainer":
        """Builds pair frequency counts across all current segmentations."""
        self.pair_counts = self._compute_pair_counts()
        self._pairs_dirty = False
        return self

    def fit(self, n_merges: int, progress: bool = True) -> "BPETrainer":
        """Main training loop: repeatedly merge the most frequent pair and update state."""
        for i in range(n_merges):
            if self._pairs_dirty:
                self.compute_pair_stats()
            if not self.pair_counts:
                break
            best_pair, freq = self.pair_counts.most_common(1)[0]
            self._apply_merge(best_pair)
            self.merges.append(best_pair)
            self._pairs_dirty = True
            if progress and (i + 1) % max(1, n_merges // 10) == 0:
                # print(f"merge {i+1}/{n_merges}: {best_pair} ({freq})")
                pass
        return self

    def build_tokenizer(self) -> BPETokenizer:
        """Freeze training artifacts and return a runtime tokenizer."""
        # Build final vocab from current symbols + merged symbols + special tokens
        symbols = set(sym for seq in self.segmented_words.values() for sym in seq)
        # Keep existing indices if you want stable ids; here we rebuild simply
        vocab = tuple(sorted(symbols))
        # Optionally prepend special tokens as dedicated IDs; keep them as strings (runtime can map)
        return BPETokenizer(
            merges=tuple(self.merges),
            vocab=vocab,
            special_tokens=self.special_tokens,
            byte_level=self.byte_level,
        )

    # ---------- Private helpers (pure-ish) ----------

    def _compute_pair_counts(self) -> Counter:
        counts: Counter[Tuple[bytes, bytes]] = Counter()
        for w, freq in self.word_counts.items():
            seq = self.segmented_words[w]
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += freq
        return counts

    def _apply_merge(self, pair: Tuple[bytes, bytes]) -> None:
        """
        Update all segmentations in-place by fusing occurrences of 'pair' into a single symbol.
        Also updates vocab_index incrementally.
        """
        merged_symbol = pair[0] + pair[1]  # concat bytes to represent the new symbol
        if merged_symbol not in self.vocab_index:
            self.vocab_index[merged_symbol] = len(self.vocab_index)

        # Incremental update of segmentations (touch only affected words)
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

        # After in-place updates, pair counts are now stale
        # We mark dirty and let compute_pair_stats rebuild, or implement a local delta update.
        # For simplicity in boilerplate: rebuild later.
        self._pairs_dirty = True

    # ---------- Optional utilities ----------

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "word_counts": dict(self.word_counts),
            "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in self.merges],
            "special_tokens": list(self.special_tokens),
            "byte_level": self.byte_level,
            "lowercase": self.lowercase,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load_checkpoint(path: str) -> "BPETrainer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        trainer = BPETrainer(
            special_tokens=obj["special_tokens"],
            byte_level=obj["byte_level"],
            lowercase=obj["lowercase"],
        )
        trainer.word_counts = Counter(obj["word_counts"])
        trainer.merges = [(a.encode("latin1"), b.encode("latin1")) for a, b in obj["merges"]]
        # segmented_words and pair_counts would need rebuild from word_counts and merges if desired
        return trainer

# Main
if __name__ == "__main__":
    with open("../data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        corpus = f.readlines()
    trainer = BPETrainer()
    trainer.pretokenize(corpus).compute_pair_stats().fit(n_merges=50_000)

    tokenizer = trainer.build_tokenizer()
