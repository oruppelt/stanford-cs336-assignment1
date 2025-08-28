import os
from typing import BinaryIO

import regex as re
from collections import Counter

from concurrent.futures import ProcessPoolExecutor
from cs336_basics.bpe_trainer import BPETrainer, build_token_pattern

input_path = "../data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 1_000
special_tokens = ["<|endoftext|>"]
num_processes = 4
_DELIM_B = b"<|endoftext|>"

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_slice(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start)

def parallel_counts(input_path, boundaries, specials_b: set[bytes]) -> Counter[bytes]:
    totals = Counter()
    with ProcessPoolExecutor() as ex:
        futs = [
            ex.submit(count_chunk, read_slice(input_path, s, e), specials_b)
            for s, e in zip(boundaries[:-1], boundaries[1:])
        ]
        for fu in futs:
            totals += fu.result()
    return totals


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def init_from_counts(trainer, word_counts: Counter[bytes]):
    trainer.word_counts = word_counts
    trainer.segmented_words = {w: [bytes([b]) for b in w] for w in word_counts}
    # Initialize full 256-byte base
    base_symbols = [bytes([b]) for b in range(256)]
    trainer.vocab_index = {sym: i for i, sym in enumerate(base_symbols)}
    trainer._pairs_dirty = True

def count_chunk_from_file(path: str, start: int, end: int, special_tokens: list[str]) -> Counter[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pat = build_token_pattern(special_tokens)
    c = Counter()
    for m in pat.finditer(chunk):
        tok = m.group(0)
        if tok:
            c[tok.encode("utf-8")] += 1
    return c

if __name__ == "__main__":
    num_processes = 4
    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    specials_b = {s.encode("utf-8") for s in special_tokens}

    totals = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futures = [
            ex.submit(count_chunk_from_file, input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        for fut in futures:
            totals += fut.result()

    # totals: Counter[bytes] from parallel pretokenization
    trainer = BPETrainer(byte_level=True)
    trainer.special_tokens = special_tokens

    # Inject parallel counts
    trainer.word_counts = totals
    trainer.segmented_words = {
        w: ([w] if w.decode("utf-8", errors="ignore") in set(special_tokens)
            else [bytes([b]) for b in w])
        for w in trainer.word_counts
    }
    # Initialize base 256-byte vocab
    trainer.vocab_index = {bytes([b]): b for b in range(256)}
    trainer._pairs_dirty = True

    # Continue training as normal
    trainer.compute_pair_stats().fit_to_vocab_size(vocab_size, special_tokens)
    vocab, merges = trainer.export_vocab_and_merges(special_tokens, vocab_size)
