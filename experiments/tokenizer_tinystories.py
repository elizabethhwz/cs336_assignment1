from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TOKENIZER_DIR = REPO_ROOT / "trained_data"
OUTPUT_DIR = REPO_ROOT / "trained_data"

TRAIN_TEXT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = TOKENIZER_DIR / "tinystories_vocab.json"
MERGES_PATH = TOKENIZER_DIR / "tinystories_merges.txt"
SPECIAL_TOKENS_PATH = TOKENIZER_DIR / "tinystories_special_tokens.txt"

TRAIN_TOKENS_PATH = OUTPUT_DIR / "tinystories_train_tokens.bin"
VALID_TOKENS_PATH = OUTPUT_DIR / "tinystories_valid_tokens.bin"
METADATA_PATH = OUTPUT_DIR / "tinystories_tokenized_metadata.json"


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    token_array = np.asarray(tokens, dtype=np.uint16)
    memmap_array = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=token_array.shape)
    memmap_array[:] = token_array[:]
    memmap_array.flush()

    print(f"encoded {input_path.name} -> {output_path}")
    print(f"num tokens: {len(token_array)}")
    print(f"dtype: {token_array.dtype}")
    return len(token_array)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(VOCAB_PATH),
        merges_filepath=str(MERGES_PATH),
        special_tokens_path=str(SPECIAL_TOKENS_PATH),
    )

    train_num_tokens = encode_file(tokenizer, TRAIN_TEXT_PATH, TRAIN_TOKENS_PATH)
    valid_num_tokens = encode_file(tokenizer, VALID_TEXT_PATH, VALID_TOKENS_PATH)

    metadata = {
        "dtype": "uint16",
        "train_tokens_path": str(TRAIN_TOKENS_PATH),
        "valid_tokens_path": str(VALID_TOKENS_PATH),
        "train_num_tokens": train_num_tokens,
        "valid_num_tokens": valid_num_tokens,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
