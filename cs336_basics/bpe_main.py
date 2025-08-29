import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Set, Iterator

import regex as re
import numpy as np
import time

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _specials_union_pattern(special_tokens: List[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    # Sort by length (longest first) to prioritize longer matches - this is added after failed test test_overlapping_special_tokens
    # Not sure if this is a good fix or just a patch.
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_tokens]
    return re.compile("|".join(escaped))


class BPETokenizer:
    """
    A simple BPE tokenizer that uses a vocabulary and merges to tokenize text.
    """
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # OLD: self.merge_rules = {pair: i for i, pair in enumerate(merges)}

        # NEW: Pre-sort merges by priority (training order) for faster lookup
        self.merge_priorities = {}
        for i, pair in enumerate(merges):
            self.merge_priorities[pair] = i

        self.token_to_id = {token: id for id, token in self.vocab.items()}

        # Precompile special token pattern for efficiency
        self.special_pattern = _specials_union_pattern(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] = None):
        """
        Load BPE tokenizer from vocab and merges files.
        """
        print(f"Loading vocab from {vocab_path}...")
        load_start = time.time()

        # Load vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        # Convert vocab back to bytes (assuming latin1 encoding was used)
        vocab = {int(id): token.encode('latin1') for id, token in vocab_json.items()}

        print(f"Vocab loaded in {time.time() - load_start:.2f}s - {len(vocab)} entries")

        # Load merges
        merge_start = time.time()
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)

        # Convert merges back to bytes tuples
        merges = [(a.encode('latin1'), b.encode('latin1')) for a, b in merges_json]

        print(f"Merges loaded in {time.time() - merge_start:.2f}s - {len(merges)} rules")

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using BPE.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        # Step 1: Split on special tokens first
        if self.special_pattern:
            segments = re.split(f'({self.special_pattern.pattern})', text)
        else:
            segments = [text]

        result_tokens = []

        for segment in segments:
            if not segment:
                continue

            # Step 2: Check if segment is a special token
            if segment in self.special_tokens:
                # Add special token directly (should be in vocab)
                special_bytes = segment.encode('utf-8')
                if special_bytes in self.token_to_id:
                    result_tokens.append(special_bytes)
                continue

            # Step 3: Apply GPT-2 pretokenization to regular text segments
            for match in PAT.finditer(segment):
                word = match.group(0)
                word_bytes = word.encode('utf-8')

                # Step 4: Apply BPE to each pretokenized word
                bpe_tokens = self._apply_bpe_optimized(word_bytes)
                result_tokens.extend(bpe_tokens)

        # Step 5: Convert tokens to IDs
        token_ids = [self.token_to_id[token] for token in result_tokens if token in self.token_to_id]
        return token_ids

    # OLD encode_batch method - keeping for compatibility
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode multiple texts. Prepared for future parallelization.

        Args:
            texts: List of input text strings

        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]

    # OLD encode_large_text method - commented out, replaced with optimized version
    # def encode_large_text(self, text: str, chunk_size: int = 1000000) -> list[int]:
    #     """
    #     Encode very large text by processing in chunks.
    #     Prepared for future parallelization.

    #     Args:
    #         text: Large input text
    #         chunk_size: Size of chunks to process

    #     Returns:
    #         List of token IDs
    #     """
    #     if len(text) <= chunk_size:
    #         return self.encode(text)

    #     # Split text into chunks at word boundaries to avoid splitting tokens
    #     chunks = []
    #     start = 0

    #     while start < len(text):
    #         end = min(start + chunk_size, len(text))

    #         # If not at the end, find a good break point (space or newline)
    #         if end < len(text):
    #             # Look backwards for a space or newline
    #             for i in range(end, max(start, end - 1000), -1):
    #                 if text[i] in ' \n\t':
    #                     end = i + 1
    #                     break

    #         chunks.append(text[start:end])
    #         start = end

    #     # Process chunks sequentially (can be parallelized later)
    #     all_token_ids = []
    #     for chunk in chunks:
    #         chunk_ids = self.encode(chunk)
    #         all_token_ids.extend(chunk_ids)

    #     return all_token_ids

    # NEW: Optimized large text encoding
    def encode_large_text_optimized(self, text: str, chunk_size: int = 2000000) -> list[int]:
        """
        Optimized encoding for very large text files.
        Pre-tokenizes entire text first, then applies BPE in batches.
        """
        print(f"Encoding large text ({len(text):,} chars) with optimized method...")

        # Step 1: Pre-tokenize entire text at once
        pretok_start = time.time()
        all_pretokens = []

        # Handle special tokens first
        if self.special_pattern:
            segments = re.split(f'({self.special_pattern.pattern})', text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens:
                special_bytes = segment.encode('utf-8')
                if special_bytes in self.token_to_id:
                    all_pretokens.append(special_bytes)
            else:
                # Batch regex matching for regular segments
                matches = list(PAT.finditer(segment))
                for match in matches:
                    word = match.group(0)
                    all_pretokens.append(word.encode('utf-8'))

        print(f"Pre-tokenization: {len(all_pretokens):,} tokens in {time.time() - pretok_start:.2f}s")

        # Step 2: Apply BPE to pretokens in batches
        bpe_start = time.time()
        all_token_ids = []
        batch_size = 10000  # Process pretokens in batches

        for i in range(0, len(all_pretokens), batch_size):
            batch = all_pretokens[i:i + batch_size]
            batch_tokens = []

            for pretoken in batch:
                if pretoken in self.token_to_id:
                    # Already a known token (special or single byte)
                    batch_tokens.append(pretoken)
                else:
                    # Apply BPE
                    bpe_tokens = self._apply_bpe_optimized(pretoken)
                    batch_tokens.extend(bpe_tokens)

            # Convert batch to IDs
            batch_ids = [self.token_to_id[token] for token in batch_tokens if token in self.token_to_id]
            all_token_ids.extend(batch_ids)

            if (i // batch_size + 1) % 5000 == 0:
                elapsed = time.time() - bpe_start
                progress = (i + len(batch)) / len(all_pretokens)
                print(f"  BPE progress: {progress:.1%} ({elapsed:.1f}s)")

        print(f"BPE encoding: {len(all_token_ids):,} final tokens in {time.time() - bpe_start:.2f}s")
        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings, yielding token IDs.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Convert IDs to token bytes
        token_bytes = []
        for token_id in ids:
            if token_id in self.vocab:
                token_bytes.append(self.vocab[token_id])

        # Concatenate all token bytes
        full_bytes = b''.join(token_bytes)

        # Decode to string, handling potential encoding errors
        try:
            return full_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Use replacement character for invalid sequences
            return full_bytes.decode('utf-8', errors='replace')

    # OLD _apply_bpe method - commented out, replaced with optimized version
    # def _apply_bpe(self, word_bytes):
    #     """Apply BPE merges to a single word."""
    #     if len(word_bytes) <= 1:
    #         return [word_bytes]

    #     # Convert to list of individual byte tokens
    #     tokens = [bytes([b]) for b in word_bytes]

    #     while len(tokens) > 1:
    #         # Get all adjacent pairs
    #         pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    #         # Find the earliest merge rule that applies
    #         best_pair = min(pairs, key=lambda p: self.merge_rules.get(p, float('inf')))

    #         # If no valid merge found, stop
    #         if self.merge_rules.get(best_pair, float('inf')) == float('inf'):
    #             break

    #         # Apply the merge
    #         a, b = best_pair
    #         new_tokens = []
    #         i = 0
    #         while i < len(tokens):
    #             if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
    #                 new_tokens.append(a + b)  # Merge the pair
    #                 i += 2
    #             else:
    #                 new_tokens.append(tokens[i])
    #                 i += 1

    #         tokens = new_tokens

    #     return tokens

    # NEW: Optimized BPE application with better algorithm
    def _apply_bpe_optimized(self, word_bytes: bytes) -> list[bytes]:
        """Optimized BPE merging with early termination and better pair finding."""
        if len(word_bytes) <= 1:
            return [word_bytes]

        # Convert to list of individual byte tokens
        tokens = [bytes([b]) for b in word_bytes]

        # Keep applying merges until no more are possible
        while len(tokens) > 1:
            # Find all possible pairs and their priorities
            best_pair = None
            best_priority = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priorities:
                    priority = self.merge_priorities[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair

            # If no valid merge found, stop
            if best_pair is None:
                break

            # Apply the merge more efficiently
            a, b = best_pair
            merged = a + b
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens


# Usage example and testing
if __name__ == "__main__":
    print("Starting optimized BPE tokenizer...")
    total_start = time.time()

    tokenizer = BPETokenizer.from_files(
        vocab_path="../artifacts/ts_train/vocab.json",
        merges_path="../artifacts/ts_train/merges.json",
        special_tokens=["<|endoftext|>"]
    )

    output_path = "../artifacts/ts_train/train_tokens.npy"

    # Test with small text
    test_start = time.time()
    test_text = "Hello world!<|endoftext|>"
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    print(f"Test encoding took {time.time() - test_start:.2f}s")

    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")

    try:
        # NEW: More efficient file reading
        file_start = time.time()
        with open("../data/TinyStoriesV2-GPT4-train.txt", "rb") as f:  # Read as binary first
            raw_data = f.read()
        full_text = raw_data.decode('utf-8', errors='replace')  # Decode with error handling
        print(f"File read took {time.time() - file_start:.2f}s - {len(full_text):,} chars")

        sample_text = full_text[:1000]  # First 1000 chars for quick test
        sample_start = time.time()
        sample_ids = tokenizer.encode(sample_text)
        print(f"Sample encoding took {time.time() - sample_start:.2f}s - {len(sample_ids)} tokens")

        # NEW: Use optimized encoding for large files
        encoding_start = time.time()
        if len(full_text) > 1000000:  # 1MB threshold
            print("Using optimized large text encoding...")
            all_ids = tokenizer.encode_large_text_optimized(full_text)
        else:
            print("Using regular encoding...")
            all_ids = tokenizer.encode(full_text)

        encoding_time = time.time() - encoding_start
        print(f"Total encoding took {encoding_time:.2f}s")

        # NEW: More efficient numpy array creation and saving
        save_start = time.time()
        # Check max token ID to determine appropriate dtype
        max_token = max(all_ids) if all_ids else 0
        if max_token < 65536:
            dtype = np.uint16
        else:
            dtype = np.uint32

        print(f"Max token ID: {max_token}, using dtype: {dtype}")
        token_array = np.array(all_ids, dtype=dtype)
        np.save(output_path, token_array)
        save_time = time.time() - save_start
        print(f"Array creation and save took {save_time:.2f}s")

        # Statistics
        total_time = time.time() - total_start
        original_bytes = len(full_text.encode('utf-8'))
        token_count = len(all_ids)

        if token_count > 0:
            bytes_per_token = original_bytes / token_count
            compression_ratio = original_bytes / token_count
            tokens_per_sec = token_count / encoding_time if encoding_time > 0 else 0

            print("\nFinal Statistics:")
            print(f"Original size: {original_bytes:,} bytes")
            print(f"Token count: {token_count:,} tokens")
            print(f"Bytes per token: {bytes_per_token:.2f}")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            print(f"Encoding speed: {tokens_per_sec:.0f} tokens/second")
            print(f"Total time: {total_time:.2f}s")
            print(f"Saved to: {output_path}")

    except FileNotFoundError:
        print("Test file not found, skipping file processing test")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
