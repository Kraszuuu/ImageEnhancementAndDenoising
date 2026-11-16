"""Train the compact U-Net brightener on paired low-light images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.models import build_model, count_parameters  # noqa: E402
from lowlight.training.brightening_datasets import (  # noqa: E402
    TRACKS,
    BrighteningDataset,
    BrighteningRow,
    d_fixed_cache_path,
    load_brightening_rows,
    select_evenly,
)
from lowlight.training.brightening_trainer import BrighteningTrainer  # noqa: E402
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.training.losses import build_loss  # noqa: E402
from lowlight.training.schedulers import build_scheduler  # noqa: E402
from lowlight.training.seed import seed_everything  # noqa: E402
from lowlight.utils.metadata import (  # noqa: E402
    environment,
    git_state,
    sha256_file,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "training" / "brightening.yaml"
    )
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss-name", default=None)
    parser.add_argument("--lambda-lpips", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="implementation check: small sample, center crops, no augmentation",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def limit_rows(rows: list[BrighteningRow], limit: int) -> list[BrighteningRow]:
    if limit <= 0:
        return rows
    datasets = sorted({row.dataset for row in rows})
    per_dataset = max(1, math.ceil(limit / len(datasets)))
    return select_evenly(rows, per_dataset)[:limit]


def verify_cache(
    rows: list[BrighteningRow], cache_dir: Path, d_fixed_path: Path
) -> dict[str, object]:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing D_fixed cache manifest: {manifest_path}; run scripts/precomputeDFixed.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = sha256_file(d_fixed_path)
    if manifest.get("d_fixed_config_sha256") != expected:
        raise SystemExit("D_fixed cache was built from a different config")
    missing = [
        row.image_id for row in rows if not d_fixed_cache_path(cache_dir, row.image_id).exists()
    ]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} D_fixed cache files, e.g. {missing[0]}; "
            "run scripts/precomputeDFixed.py"
        )
    return manifest


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    defaults = load_yaml(ROOT / "configs" / "training" / "gpu_defaults.yaml")
    model_config = load_yaml(ROOT / str(config["model_config"]))
    training = config["training"]
    data = config["data"]

    seed = args.seed
    tag = args.tag or f"seed{seed}"
    out_dir = Path(
        args.out_dir or ROOT / "results" / "brightening_cnn" / args.track / tag
    ).resolve()
    checkpoint_dir = Path(
        args.checkpoint_dir or ROOT / "checkpoints" / "brightening_cnn" / args.track / tag
    ).resolve()
    device = args.device or str(defaults.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing CPU target training")
    epochs = int(args.epochs or training["epochs"])
    patience = int(args.patience or training["patience"])
    lr = float(args.lr or training["lr"])
    loss_name = str(args.loss_name or config["loss"]["name"])
    lambda_lpips = float(args.lambda_lpips or config["loss"]["lambda_lpips"])
    num_workers = int(
        args.num_workers if args.num_workers is not None else defaults.get("num_workers", 4)
    )

    seed_everything(seed, deterministic=bool(defaults.get("deterministic", True)))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_brightening_rows(ROOT, [ROOT / str(path) for path in data["train_splits"]])
    val_rows = load_brightening_rows(ROOT, [ROOT / str(path) for path in data["val_splits"]])
    if args.overfit:
        args.limit_train = args.limit_train or 8
        args.limit_val = args.limit_val or args.limit_train
        val_rows = train_rows
    train_rows = limit_rows(train_rows, args.limit_train)
    val_max = int(data["val_max_images_per_dataset"])
    if args.limit_val:
        val_rows = limit_rows(val_rows, args.limit_val)
    elif val_max > 0:
        val_rows = select_evenly(val_rows, val_max)
    if not train_rows or not val_rows:
        raise SystemExit("empty train or validation split")

    d_fixed_path = ROOT / str(data["d_fixed_config"])
    cache_dir = ROOT / str(data["d_fixed_cache"])
    cache_manifest = None
    if args.track == "d_fixed":
        cache_manifest = verify_cache(train_rows + val_rows, cache_dir, d_fixed_path)

    patch_size = int(defaults["patch_size"])
    train_dataset = BrighteningDataset(
        train_rows,
        track=args.track,
        training=True,
        cache_dir=cache_dir if args.track == "d_fixed" else None,
        seed=seed,
        patch_size=patch_size,
        augment=not args.overfit,
        fixed_crop=args.overfit,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(defaults["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )
    val_dataset = BrighteningDataset(
        val_rows,
        track=args.track,
        training=False,
        cache_dir=cache_dir if args.track == "d_fixed" else None,
        seed=seed,
    )
    val_samples = [val_dataset[index] for index in range(len(val_dataset))]

    model = build_model(model_config["model"]).to(device)
    loss_fn = build_loss(loss_name, lambda_lpips=lambda_lpips, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        name=str(training["scheduler"]),
        total_epochs=epochs,
        warmup_epochs=int(training["warmup_epochs"]),
        eta_min_factor=float(training["eta_min_factor"]),
    )
    checkpoint = CheckpointManager(checkpoint_dir)
    resolved = {
        "config_path": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "track": args.track,
        "overrides": {
            "seed": seed,
            "tag": tag,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "loss_name": loss_name,
            "lambda_lpips": lambda_lpips,
            "limit_train": args.limit_train,
            "limit_val": args.limit_val,
            "num_workers": num_workers,
            "overfit": args.overfit,
        },
        "data": data,
        "loss": {"name": loss_name, "lambda_lpips": lambda_lpips},
        "training": {**training, "epochs": epochs, "patience": patience, "lr": lr},
        "model": model_config["model"],
        "gpu_defaults": defaults,
    }
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    metadata: dict[str, object] = {
        "run_id": f"stage05-brightening-{args.track}-{tag}",
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "seed": seed,
        "track": args.track,
        "git": git_state(ROOT),
        "environment": environment(device),
        "deterministic": bool(defaults.get("deterministic", True)),
        "amp": bool(defaults.get("amp", True)),
        "model": {
            "name": model_config["model"].get("name", "compact_unet"),
            "config": model_config["model"],
            "parameters": count_parameters(model),
        },
        "config": resolved,
        "config_sha256": hashlib.sha256(
            json.dumps(resolved, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "d_fixed_config_sha256": sha256_file(d_fixed_path),
        "d_fixed_cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json")
        if cache_manifest is not None
        else None,
        "split_hashes": split_manifest.get("files_sha256", {}),
        "n_train_images": len(train_rows),
        "n_val_images": len(val_rows),
    }
    write_json(out_dir / "run_metadata.json", metadata)

    trainer = BrighteningTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_samples=val_samples,
        checkpoint=checkpoint,
        run_dir=out_dir,
        config=resolved,
        seed=seed,
        device=device,
        amp=bool(defaults.get("amp", True)),
        grad_clip=float(training["grad_clip"]),
        grad_accum_steps=int(defaults.get("grad_accum_steps", 1)),
        max_epochs=epochs,
        patience=patience,
        min_delta=float(training["min_delta"]),
    )
    print(
        f"track={args.track} tag={tag} seed={seed} device={device} "
        f"params={count_parameters(model)} train={len(train_rows)} "
        f"val={len(val_rows)} loss={loss_name}"
    )
    started = time.perf_counter()
    result = trainer.fit()
    duration = time.perf_counter() - started
    metadata.update(
        {
            "finished_at_utc": utc_now(),
            "duration_seconds": duration,
            "best": result["best"],
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "epochs_ran": result["epochs_ran"],
            "checkpoint": str(checkpoint.best_path.relative_to(ROOT)),
            "history": str((out_dir / "history.csv").relative_to(ROOT)),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if device.startswith("cuda")
            else None,
        }
    )
    write_json(out_dir / "run_metadata.json", metadata)
    write_json(
        out_dir / "train_summary.json",
        {
            "track": args.track,
            "tag": tag,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "epochs_ran": result["epochs_ran"],
            "duration_seconds": duration,
            "checkpoint": str(checkpoint.best_path.relative_to(ROOT)),
        },
    )
    print(
        f"best_epoch={result['best_epoch']} best_val_psnr={result['best_score']:.3f} "
        f"duration={duration:.1f}s checkpoint={checkpoint.best_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
