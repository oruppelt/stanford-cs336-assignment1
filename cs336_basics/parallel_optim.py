# Use 'regex' compiled once at module level
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from cs336_basics.bpe_optimized import OptimizedBPETrainer, train_bpe_tokenizer_optimized
import json

_BASE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(_BASE_PAT)

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    # keep delimiters so we can ignore them later deterministically if needed
    return re.split(f"({pattern})", text)

def _count_segment_batch(seg_batch: list[str], specials: set[str]) -> Counter[bytes]:
    c = Counter()
    for seg in seg_batch:
        if not seg or seg in specials:
            continue
        for m in PAT.finditer(seg):
            tok = m.group(0)
            if tok:
                c[tok.encode("utf-8")] += 1
    return c

def parallel_counts_by_segments(
    input_path: str,
    special_tokens: list[str],
    max_workers: int | None = None,
) -> Counter[bytes]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = split_by_special_tokens(text, special_tokens)
    specials = set(special_tokens)
    parts = [p for p in parts if p]  # drop empties

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Batch segments to reduce task count / IPC overhead
    target_tasks = max_workers * 8
    if len(parts) <= target_tasks:
        batches = [[p] for p in parts]
    else:
        batch_size = (len(parts) + target_tasks - 1) // target_tasks
        batches = [parts[i:i + batch_size] for i in range(0, len(parts), batch_size)]

    totals = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for c in ex.map(_count_segment_batch, batches, [specials] * len(batches)):
            totals.update(c)
    return totals


def run_train_bpe_parallel_optimized(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int | None = None
):
    """
    Run BPE training with parallel pretokenization and optimized merging.
    Best of both worlds: parallel I/O processing + fast incremental merging.
    """
    wc = parallel_counts_by_segments(input_path, special_tokens, num_processes)

    # Use optimized trainer with precomputed counts
    trainer = OptimizedBPETrainer(special_tokens=special_tokens)
    trainer.word_counts = wc
    trainer.segmented_words = {w: [bytes([b]) for b in w] for w in wc}
    trainer.vocab_index = {bytes([b]): b for b in range(256)}
    trainer.fit_to_vocab_size(vocab_size, progress=True)
    return trainer.export_vocab_and_merges(vocab_size)


if __name__ == "__main__":
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    print("Running parallel + optimized BPE training...")
    vocab, merges = run_train_bpe_parallel_optimized(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=4,
    )

    # Serialize (bytes-safe via latin1 round-trip)
    out_dir = "../artifacts/ts_train"
    os.makedirs(out_dir, exist_ok=True)

    # vocab: id -> token-bytes (as latin1 string)
    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({int(i): tok.decode("latin1") for i, tok in vocab.items()},
                  f, ensure_ascii=False, indent=2)

    # merges: list of [token1_bytes, token2_bytes] (as latin1 strings)
    merges_path = os.path.join(out_dir, "merges.json")
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump([[a.decode("latin1"), b.decode("latin1")] for (a, b) in merges],
                  f, ensure_ascii=False, indent=2)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Final sizes → vocab: {len(vocab)}  merges: {len(merges)}  specials: {len(special_tokens)}")