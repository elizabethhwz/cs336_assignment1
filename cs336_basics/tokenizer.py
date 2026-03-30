from typing import Iterable
import json
import regex as re


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initializes the tokenizer with the given vocabulary, merges, and special tokens.
        :param vocab: A dictionary mapping tokens to their corresponding IDs.
        :param merges: A list of merge operations for the BPE tokenizer.
        :param special_tokens: A list of special tokens to be included in the tokenizer.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], reverse=True)
        self.reversed_vocab = {v: k for k, v in vocab.items()}
        self.merges_priority = {merge: i for i, merge in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens_path: str | None = None,
    ):
        """
        Class method to create a Tokenizer instance from files containing the vocabulary, merges, and special tokens.
        :param vocab_filepath: Path to the file containing the vocabulary.
        :param merges_filepath: Path to the file containing the merge operations.
        :param special_tokens_path: Path to the file containing the special tokens (optional)."""
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            token_id: bytes([byte_decoder[ch] for ch in token])
            for token, token_id in raw_vocab.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = []
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                left, right = stripped.split()
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in left]),
                        bytes([byte_decoder[ch] for ch in right]),
                    )
                )

        special_tokens = []
        if special_tokens_path:
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                special_tokens = [line.strip() for line in f if line.strip()]

        for special_token in special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in set(vocab.values()):
                vocab[len(vocab)] = special_token_bytes

        return cls(vocab, merges, special_tokens)

    def split_by_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        return parts

    def find_best_merge(
        self, tokens: list[bytes]
    ) -> tuple[tuple[bytes, bytes] | None, int | None]:
        best_merge = None
        best_priority = len(self.merges)  
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges_priority and self.merges_priority[pair] < best_priority:
                best_merge = pair
                best_priority = self.merges_priority[pair]
        return best_merge, (best_priority if best_merge else None)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        :param text: The input string to encode.
        :return: A list of token IDs.
        """
        ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in self.split_by_special_tokens(text):
            encoded_text = part.encode("utf-8")
            if part in self.special_tokens:
                ids.append(self.reversed_vocab[encoded_text])
            else:
                for match in re.finditer(PAT, part):
                    originals = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                    original_tokens = list(originals)
                    while True:
                        best_merge, _ = self.find_best_merge(original_tokens)
                        if not best_merge:
                            break
                        new_token = b"".join(best_merge)
                        i = 0
                        new_tokens = []
                        while i < len(original_tokens):
                            if (
                                i < len(original_tokens) - 1
                                and (original_tokens[i], original_tokens[i + 1]) == best_merge
                            ):
                                new_tokens.append(new_token)
                                i += 2
                            else:
                                new_tokens.append(original_tokens[i])
                                i += 1
                        original_tokens = new_tokens
                    ids.extend([self.reversed_vocab[token] for token in original_tokens])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encodes an iterable of strings into an iterable of token IDs.
        :param iterable: An iterable of strings to encode.
        :return: An iterable of token IDs.
        """
        # Implement the encoding logic for a list of texts
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        :param ids: A list of token IDs to decode.
        :return: The decoded string."""
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
