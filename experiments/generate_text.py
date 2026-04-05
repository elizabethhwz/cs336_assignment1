from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.decoder import decoder
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "train_llm_config.json"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "trained_data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 language model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the training config JSON.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="Override inference device.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling threshold.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Directory containing tinystories tokenizer artifacts.",
    )
    return parser


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath=str(tokenizer_dir / "tinystories_vocab.json"),
        merges_filepath=str(tokenizer_dir / "tinystories_merges.txt"),
        special_tokens_path=str(tokenizer_dir / "tinystories_special_tokens.txt"),
    )


def build_model(config: dict, device: str) -> TransformerLM:
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    ).to(device)
    return model


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device"))
    checkpoint_path = args.checkpoint or Path(config["checkpoint_path"])

    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = build_model(config, device)
    load_model_checkpoint(model, checkpoint_path, device)

    eos_token_id = tokenizer.reversed_vocab[b"<|endoftext|>"]

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running on device: {device}")
    print("Enter a prompt. Submit an empty line to exit.\n")

    while True:
        prompt_text = input("prompt> ").strip()
        if prompt_text == "":
            break

        prompt_ids = tokenizer.encode(prompt_text)
        generated_tokens = decoder(
            model=model,
            context_length=config["context_length"],
            prompt=prompt_ids,
            eos_token_id=eos_token_id,
            p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        generated_ids = generated_tokens[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        print("\ncompletion:")
        print(generated_text)
        print()


if __name__ == "__main__":
    main()
