import json
from collections import Counter
from itertools import pairwise

import regex as re


class BPE:
    def __init__(self, vocab_size : int):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for base byte vocabulary.")
        self.vocab_size = vocab_size

        # Maps (token_id1, token_id2) -> new_token_id
        self.merges: dict[tuple[int, int], int] = {}

        # Maps token_id -> bytes
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        # Pre Normalization
        self.pattern = re.compile(
                r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
                r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""
                r"""|\p{N}{1,3}"""
                r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""
                r"""|\s*[\r\n]+"""
                r"""|\s+(?!\S)"""
                r"""|\s+"""
            )

    @staticmethod
    def _get_stats(chunks: list[list[int]]) -> Counter:
        """Counts frequency of all adjacent pairs within chunks."""
        stats = Counter()
        for chunk in chunks:
            stats.update(pairwise(chunk))
        return stats

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replaces all consecutive occurrences of pair with new_id."""
        new_ids = []
        i = 0
        while i < len(ids):
            # Check if current and next element match the target pair
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _save(self, prefix: str) -> None:
        # 1. Save vocab using .hex() to preserve exact raw bytes reversibly
        serializable_vocab = {
            token_id: byte_val.hex() for token_id, byte_val in self.vocab.items()
        }
        vocab_path = f"{prefix}.vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=2)

        # 2. convert pair to a string key for JSON
        serializable_merges = {
            f"{p0} {p1}": new_id
            for (p0, p1), new_id in sorted(self.merges.items(), key=lambda item: item[1])
        }
        merges_path = f"{prefix}.merges.json"
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f, indent=2)

        print(f"Saved: {vocab_path} and {merges_path}")
    @classmethod
    def load(cls, prefix: str):
        """Loads merges and vocab from disk and constructs a BPE instance."""
        vocab_path = f"{prefix}.vocab.json"
        merges_path = f"{prefix}.merges.json"
        # 1. Load Vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # Convert keys back to int and hex strings back to bytes
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}
        # 2. Load Merges
        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = {}
        for key_pair, new_id in raw_merges.items():
            p0_str, p1_str = key_pair.split()
            pair = (int(p0_str), int(p1_str))
            merges[pair] = int(new_id)
        # 3. Instantiate and attach loaded state
        tokenizer = cls(vocab_size=len(vocab))
        tokenizer.merges = merges
        tokenizer.vocab = vocab
        return tokenizer

    def train(self, text: str, prefix:str , verbose: bool = False):
        """Learns merges from training corpus until target vocab_size is reached."""
        raw_chunks = self.pattern.findall(text)
        chunks : list[list[int]] = [list(c.encode("utf-8")) for c in raw_chunks]
        num_merges = self.vocab_size - 256

        print(f"Starting BPE training: {num_merges} merges planned.")
        for i in range(num_merges):
            stats = BPE._get_stats(chunks)
            if not stats:
                print(f"No more pairs to merge. Stopping at merge {i}.")
                break
            # Find the most frequent pair
            best_pair = stats.most_common(1)[0][0]
            new_id = 256 + i
            # Record merge and update vocabulary
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            # Replace pair occurrences in the training stream
            chunks = [BPE._merge(chunk, best_pair, new_id) for chunk in chunks]
            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id]!r})")
        print("Training complete.")
        self._save(prefix)

    def encode(self, text: str) -> list[int]:
        """Tokenizes an input string into token IDs using learned merges."""
        # Split input into the same chunks
        raw_chunks = self.pattern.findall(text)
        encoded_ids = []
        for raw_chunk in raw_chunks:
            chunk_ids = list(raw_chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = pairwise(chunk_ids)
                pair_candidates = {pair: self.merges[pair] for pair in stats if pair in self.merges}
                if not pair_candidates:
                    break
                # Select the pair that was merged earliest during training
                best_pair = min(pair_candidates, key=pair_candidates.__getitem__)
                chunk_ids = self._merge(chunk_ids, best_pair, self.merges[best_pair])
            encoded_ids.extend(chunk_ids)
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")