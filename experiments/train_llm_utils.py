import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.data_loader import data_loader
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


REQUIRED_CONFIG_KEYS = {
    "train_data",
    "val_data",
    "checkpoint_path",
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
    "batch_size",
    "max_iters",
    "lr_max",
    "lr_min",
    "warmup_iters",
    "weight_decay",
    "beta1",
    "beta2",
    "eps",
    "grad_clip",
    "log_every",
    "eval_every",
    "eval_steps",
    "save_every",
    "seed",
    "dtype",
    "use_wandb",
    "wandb_project",
}

OPTIONAL_CONFIG_KEYS = REQUIRED_CONFIG_KEYS | {"resume_from", "device", "wandb_run_name"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train_llm_config.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. cpu, cuda, cuda:0, mps.")

    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--use-wandb",
        dest="use_wandb",
        action="store_true",
        default=None,
        help="Enable Weights and Biases logging.",
    )
    wandb_group.add_argument(
        "--no-use-wandb",
        dest="use_wandb",
        action="store_false",
        default=None,
        help="Disable Weights and Biases logging.",
    )

    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object.")
    return config


def merge_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    if cli_args.config is None or not cli_args.config.exists():
        raise FileNotFoundError(f"Config file not found: {cli_args.config}")

    merged = load_config(cli_args.config)
    unknown_keys = sorted(set(merged) - OPTIONAL_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    if cli_args.resume_from is not None:
        merged["resume_from"] = cli_args.resume_from
    if cli_args.device is not None:
        merged["device"] = cli_args.device
    if cli_args.use_wandb is not None:
        merged["use_wandb"] = cli_args.use_wandb

    missing_keys = sorted(key for key in REQUIRED_CONFIG_KEYS if key not in merged)
    if missing_keys:
        raise ValueError(f"Missing required settings: {', '.join(missing_keys)}")

    for path_key in ["train_data", "val_data", "checkpoint_path", "resume_from", "config"]:
        if merged.get(path_key) is not None:
            merged[path_key] = Path(merged[path_key])

    merged["config"] = cli_args.config
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    cli_args = parser.parse_args()
    return merge_config(cli_args)


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        dataset = np.load(path, mmap_mode="r")
    else:
        dataset = np.memmap(path, mode="r", dtype=np.dtype(dtype))
    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token array at {path}, got shape {dataset.shape}.")
    return dataset


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_steps: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = data_loader(dataset, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        losses.append(cross_entropy_loss(logits, y).item())
    model.train()
    return float(np.mean(losses))


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any | None:
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Weights and Biases logging requested, but wandb is not installed.") from exc

    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)


def namespace_to_config(args: argparse.Namespace, device: str) -> dict[str, Any]:
    return {
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "checkpoint_path": str(args.checkpoint_path),
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "device": device,
        "seed": args.seed,
        "dtype": args.dtype,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "lr_max": args.lr_max,
        "lr_min": args.lr_min,
        "warmup_iters": args.warmup_iters,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "eval_steps": args.eval_steps,
        "save_every": args.save_every,
    }


def run_training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    train_data = load_dataset(args.train_data, args.dtype)
    val_data = load_dataset(args.val_data, args.dtype)

    if len(train_data) <= args.context_length:
        raise ValueError("Training dataset must be longer than context_length.")
    if len(val_data) <= args.context_length:
        raise ValueError("Validation dataset must be longer than context_length.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    config = namespace_to_config(args, device)
    wandb_run = maybe_init_wandb(args, config)

    print(json.dumps(config, indent=2))

    model.train()
    for iteration in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            t=iteration,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            warmup_T=args.warmup_iters,
            cosine_annealing_T=args.max_iters,
        )
        set_learning_rate(optimizer, lr)

        x, y = data_loader(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if iteration % args.log_every == 0:
            train_loss = loss.item()
            print(f"iter={iteration} train_loss={train_loss:.4f} lr={lr:.6g}")
            if wandb_run is not None:
                wandb_run.log({"iter": iteration, "train/loss": train_loss, "lr": lr})

        if iteration % args.eval_every == 0 or iteration == args.max_iters - 1:
            val_loss = evaluate(
                model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_steps=args.eval_steps,
            )
            print(f"iter={iteration} val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"iter": iteration, "val/loss": val_loss})

        if ((iteration + 1) % args.save_every == 0) or (iteration == args.max_iters - 1):
            save_checkpoint(model, optimizer, iteration + 1, args.checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()
