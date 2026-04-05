from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from tests.adapters import run_train_bpe

from cs336_basics.tokenizer import gpt2_bytes_to_unicode


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "trained_data"
TRAIN_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]


def serialize_vocab(vocab: dict[int, bytes], output_path: Path) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    serializable_vocab = {
        b"".join(byte_encoder[b].encode("utf-8") for b in token_bytes).decode("utf-8"): token_id
        for token_id, token_bytes in vocab.items()
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)


def serialize_merges(merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    with output_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_text = b"".join(byte_encoder[b].encode("utf-8") for b in left).decode("utf-8")
            right_text = b"".join(byte_encoder[b].encode("utf-8") for b in right).decode("utf-8")
            f.write(f"{left_text} {right_text}\n")


def serialize_special_tokens(special_tokens: list[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(f"{token}\n")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training corpus not found: {TRAIN_PATH}")
    if not VALID_PATH.exists():
        raise FileNotFoundError(f"Validation corpus not found: {VALID_PATH}")

    start_time = perf_counter()
    vocab, merges = run_train_bpe(
        input_path=str(TRAIN_PATH),
        vocab_size=10000,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed_time = perf_counter() - start_time

    vocab_path = OUT_DIR / "tinystories_vocab.json"
    merges_path = OUT_DIR / "tinystories_merges.txt"
    special_tokens_path = OUT_DIR / "tinystories_special_tokens.txt"
    metadata_path = OUT_DIR / "tinystories_tokenizer_metadata.json"

    serialize_vocab(vocab, vocab_path)
    serialize_merges(merges, merges_path)
    serialize_special_tokens(SPECIAL_TOKENS, special_tokens_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_text_path": str(TRAIN_PATH),
                "valid_text_path": str(VALID_PATH),
                "vocab_path": str(vocab_path),
                "merges_path": str(merges_path),
                "special_tokens_path": str(special_tokens_path),
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
            indent=2,
        )

    print("vocab size:", len(vocab))
    print("num merges:", len(merges))
    print("first 10 merges:", merges[:10])
    print("train text:", TRAIN_PATH)
    print("valid text:", VALID_PATH)
    print("vocab file:", vocab_path)
    print("merges file:", merges_path)
    print("special tokens file:", special_tokens_path)
    print("metadata file:", metadata_path)
    print(f"execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
