import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.data_loader import data_loader
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM
from experiments.train_llm_utils import load_config, load_dataset, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a learning-rate sweep for TinyStories training.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("experiments/train_llm_config.json"),
        help="Base JSON config to copy for each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lr_sweep_results"),
        help="Directory where run histories, checkpoints, and plots are written.",
    )
    parser.add_argument(
        "--lr-values",
        type=float,
        nargs="+",
        required=True,
        help="Learning rates to try for lr_max. lr_min keeps the same ratio as the base config unless overridden.",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device override.")
    parser.add_argument("--max-iters", type=int, default=None, help="Optional override for max_iters.")
    parser.add_argument("--eval-every", type=int, default=None, help="Optional override for eval_every.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Optional override for eval_steps.")
    parser.add_argument("--save-every", type=int, default=None, help="Optional override for save_every.")
    parser.add_argument("--log-every", type=int, default=None, help="Optional override for log_every.")
    parser.add_argument("--warmup-iters", type=int, default=None, help="Optional override for warmup_iters.")
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=None,
        help="If set, use lr_min = lr_max * lr_min_ratio for every run.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for faster startup during short sweeps.",
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
        default="lr-grid-search",
        help="W&B group name used to collect all runs in this sweep.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--write-local-plot",
        action="store_true",
        help="Also write a local SVG plot. Off by default because W&B already provides learning curves.",
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
    lr_value: float,
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

    run_name = f"lr_{lr_value:.0e}".replace("+", "")
    checkpoint_path = output_dir / f"{run_name}.pt"
    history_path = output_dir / f"{run_name}_history.csv"

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
                "checkpoint_path": str(checkpoint_path),
                "history_path": str(history_path),
            },
            reinit=True,
        )

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["iteration", "split", "loss", "lr"])
        writer.writeheader()

        progress_bar = tqdm(
            range(config["max_iters"]),
            desc=f"LR {lr_value:.0e}",
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
                writer.writerow(record)
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
                writer.writerow(record)
                history.append(record)
                if wandb_run is not None:
                    wandb_run.log({"iter": iteration, "val/loss": latest_val_loss, "lr": lr})

            progress_bar.set_postfix_str(f"train={train_loss:.4f} val={latest_val_loss:.4f} lr={lr:.2e}")

        progress_bar.close()

    if not diverged:
        save_checkpoint(model, optimizer, config["max_iters"], checkpoint_path)

    val_points = [row for row in history if row["split"] == "val"]
    train_points = [row for row in history if row["split"] == "train"]
    final_val_loss = float(val_points[-1]["loss"]) if val_points else float("nan")
    final_train_loss = float(train_points[-1]["loss"]) if train_points else float("nan")

    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = final_train_loss
        wandb_run.summary["final_val_loss"] = final_val_loss
        wandb_run.summary["diverged"] = diverged
        wandb_run.summary["checkpoint_path"] = str(checkpoint_path) if not diverged else ""
        wandb_run.summary["history_path"] = str(history_path)
        wandb_run.finish()

    return {
        "run_name": run_name,
        "lr_max": config["lr_max"],
        "lr_min": config["lr_min"],
        "max_iters": config["max_iters"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "diverged": diverged,
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path) if not diverged else "",
    }


def plot_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    series: list[dict[str, Any]] = []
    max_iteration = 0
    min_loss = float("inf")
    max_loss = float("-inf")

    for result in results:
        history_path = Path(result["history_path"])
        points: list[tuple[int, float]] = []
        with history_path.open("r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row["split"] != "val":
                    continue
                iteration = int(row["iteration"])
                loss = float(row["loss"])
                points.append((iteration, loss))
                max_iteration = max(max_iteration, iteration)
                min_loss = min(min_loss, loss)
                max_loss = max(max_loss, loss)
        if points:
            series.append({"label": f"lr={result['lr_max']:.0e}", "diverged": result["diverged"], "points": points})

    if not series:
        return

    width = 900
    height = 600
    margin_left = 80
    margin_right = 30
    margin_top = 40
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#0f766e", "#dc2626", "#2563eb", "#ca8a04", "#7c3aed", "#ea580c"]

    if min_loss == max_loss:
        min_loss -= 0.5
        max_loss += 0.5

    def x_pos(iteration: int) -> float:
        if max_iteration == 0:
            return margin_left
        return margin_left + (iteration / max_iteration) * plot_width

    def y_pos(loss: float) -> float:
        return margin_top + ((max_loss - loss) / (max_loss - min_loss)) * plot_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="Arial">TinyStories validation curves across learning rates</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#444" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#444" stroke-width="1.5"/>',
    ]

    for tick_ratio in np.linspace(0.0, 1.0, num=6):
        tick_loss = min_loss + (1.0 - tick_ratio) * (max_loss - min_loss)
        y = margin_top + tick_ratio * plot_height
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{tick_loss:.2f}</text>'
        )

    for tick_ratio in np.linspace(0.0, 1.0, num=6):
        iteration = int(round(tick_ratio * max_iteration))
        x = margin_left + tick_ratio * plot_width
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial">{iteration}</text>'
        )

    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        path = " ".join(
            f"{'M' if point_index == 0 else 'L'} {x_pos(iteration):.2f} {y_pos(loss):.2f}"
            for point_index, (iteration, loss) in enumerate(item["points"])
        )
        dash = ' stroke-dasharray="8 6"' if item["diverged"] else ""
        svg_parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
        for iteration, loss in item["points"]:
            svg_parts.append(
                f'<circle cx="{x_pos(iteration):.2f}" cy="{y_pos(loss):.2f}" r="4" fill="{color}"/>'
            )

    legend_x = margin_left
    legend_y = height - 25
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        x = legend_x + index * 135
        dash = ' stroke-dasharray="8 6"' if item["diverged"] else ""
        svg_parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash}/>')
        label = item["label"] + (" diverged" if item["diverged"] else "")
        svg_parts.append(
            f'<text x="{x + 30}" y="{legend_y + 4}" font-size="12" font-family="Arial">{label}</text>'
        )

    svg_parts.append(
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="14" font-family="Arial">Iteration</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height / 2})">Validation loss</text>'
    )
    svg_parts.append("</svg>")

    (output_dir / "lr_sweep_curves.svg").write_text("\n".join(svg_parts), encoding="utf-8")


def apply_overrides(base_config: dict[str, Any], args: argparse.Namespace, lr_value: float) -> dict[str, Any]:
    config = dict(base_config)
    base_ratio = base_config["lr_min"] / base_config["lr_max"]

    config["lr_max"] = lr_value
    config["lr_min"] = lr_value * (args.lr_min_ratio if args.lr_min_ratio is not None else base_ratio)

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
    for lr_value in args.lr_values:
        run_config = apply_overrides(base_config, args, lr_value)
        result = run_single_sweep(
            config=run_config,
            device=device,
            output_dir=output_dir,
            lr_value=lr_value,
            no_compile=args.no_compile,
            use_wandb=args.use_wandb,
            wandb_project=wandb_project,
            wandb_group=args.wandb_group,
            wandb_entity=args.wandb_entity,
        )
        results.append(result)

    if args.write_local_plot:
        plot_results(output_dir, results)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved sweep summary to {summary_path}")
    if args.write_local_plot:
        print(f"Saved validation plot to {output_dir / 'lr_sweep_curves.svg'}")


if __name__ == "__main__":
    main()
