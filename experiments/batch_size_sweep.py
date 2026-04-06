import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.data_loader import data_loader
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM
from experiments.train_llm_utils import load_config, load_dataset, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch-size sweep while holding learning rate fixed.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("experiments/train_llm_config.json"),
        help="Base JSON config to copy for each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/batch_size_sweep_results"),
        help="Directory where the final sweep summary is written.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Batch sizes to try.",
    )
    parser.add_argument(
        "--fixed-lr",
        type=float,
        default=8e-4,
        help="Learning rate used for lr_max during the batch-size sweep.",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=None,
        help="If set, use lr_min = fixed_lr * lr_min_ratio. Otherwise keep the base-config ratio.",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device override.")
    parser.add_argument("--max-iters", type=int, default=None, help="Optional override for max_iters.")
    parser.add_argument("--eval-every", type=int, default=None, help="Optional override for eval_every.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Optional override for eval_steps.")
    parser.add_argument("--save-every", type=int, default=None, help="Optional override for save_every.")
    parser.add_argument("--log-every", type=int, default=None, help="Optional override for log_every.")
    parser.add_argument("--warmup-iters", type=int, default=None, help="Optional override for warmup_iters.")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile and run the model in eager mode.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log each trial as a separate Weights & Biases run.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. Defaults to the base config project when available.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="batch-size-grid-search",
        help="W&B group name used to collect all runs in this sweep.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    return parser


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


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def maybe_compile(model: torch.nn.Module, device: str, no_compile: bool) -> torch.nn.Module:
    if no_compile:
        return model
    if device == "cpu":
        return torch.compile(model)
    if device == "mps":
        return torch.compile(model, backend="aot_eager")
    return model


def run_single_sweep(
    config: dict[str, Any],
    device: str,
    output_dir: Path,
    batch_size: int,
    no_compile: bool,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_group: str | None,
    wandb_entity: str | None,
) -> dict[str, Any]:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_data = load_dataset(Path(config["train_data"]), config["dtype"])
    val_data = load_dataset(Path(config["val_data"]), config["dtype"])

    run_name = f"bs_{batch_size}"

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    ).to(device)
    model = maybe_compile(model, device, no_compile)

    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=config["lr_max"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
        eps=config["eps"],
    )

    history: list[dict[str, float | int | str]] = []
    latest_val_loss = float("nan")
    diverged = False
    wandb_run = None

    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=run_name,
            config={
                **config,
                "device": device,
            },
            reinit=True,
        )

    progress_bar = tqdm(
        range(config["max_iters"]),
        desc=f"BS {batch_size}",
        dynamic_ncols=True,
        unit="iter",
    )

    for iteration in progress_bar:
        lr = get_lr_cosine_schedule(
            t=iteration,
            lr_max=config["lr_max"],
            lr_min=config["lr_min"],
            warmup_T=config["warmup_iters"],
            cosine_annealing_T=config["max_iters"],
        )
        set_learning_rate(optimizer, lr)

        x, y = data_loader(
            train_data,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=device,
        )
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        train_loss = float(loss.item())

        if not np.isfinite(train_loss):
            diverged = True
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=config["grad_clip"])
        optimizer.step()

        if iteration % config["log_every"] == 0 or iteration == config["max_iters"] - 1:
            record = {"iteration": iteration, "split": "train", "loss": train_loss, "lr": lr}
            history.append(record)
            if wandb_run is not None:
                wandb_run.log({"iter": iteration, "train/loss": train_loss, "lr": lr})

        if iteration % config["eval_every"] == 0 or iteration == config["max_iters"] - 1:
            latest_val_loss = evaluate(
                model,
                dataset=val_data,
                batch_size=config["batch_size"],
                context_length=config["context_length"],
                device=device,
                eval_steps=config["eval_steps"],
            )
            if not np.isfinite(latest_val_loss):
                diverged = True
                break
            record = {"iteration": iteration, "split": "val", "loss": latest_val_loss, "lr": lr}
            history.append(record)
            if wandb_run is not None:
                wandb_run.log({"iter": iteration, "val/loss": latest_val_loss, "lr": lr})

        progress_bar.set_postfix_str(
            f"train={train_loss:.4f} val={latest_val_loss:.4f} lr={lr:.2e} bs={config['batch_size']}"
        )

    progress_bar.close()

    val_points = [row for row in history if row["split"] == "val"]
    train_points = [row for row in history if row["split"] == "train"]
    final_val_loss = float(val_points[-1]["loss"]) if val_points else float("nan")
    final_train_loss = float(train_points[-1]["loss"]) if train_points else float("nan")

    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = final_train_loss
        wandb_run.summary["final_val_loss"] = final_val_loss
        wandb_run.summary["diverged"] = diverged
        wandb_run.finish()

    return {
        "run_name": run_name,
        "batch_size": config["batch_size"],
        "lr_max": config["lr_max"],
        "lr_min": config["lr_min"],
        "max_iters": config["max_iters"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "diverged": diverged,
    }


def apply_overrides(base_config: dict[str, Any], args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    config = dict(base_config)
    base_ratio = base_config["lr_min"] / base_config["lr_max"]

    config["batch_size"] = batch_size
    config["lr_max"] = args.fixed_lr
    config["lr_min"] = args.fixed_lr * (args.lr_min_ratio if args.lr_min_ratio is not None else base_ratio)

    for key in ["max_iters", "eval_every", "eval_steps", "save_every", "log_every", "warmup_iters"]:
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    device = resolve_device(args.device or base_config.get("device"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_project = args.wandb_project or base_config.get("wandb_project")

    results = []
    for batch_size in args.batch_sizes:
        run_config = apply_overrides(base_config, args, batch_size)
        result = run_single_sweep(
            config=run_config,
            device=device,
            output_dir=output_dir,
            batch_size=batch_size,
            no_compile=args.no_compile,
            use_wandb=args.use_wandb,
            wandb_project=wandb_project,
            wandb_group=args.wandb_group,
            wandb_entity=args.wandb_entity,
        )
        results.append(result)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()