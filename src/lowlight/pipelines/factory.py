"""Build combined pipelines from frozen YAML configs."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from lowlight.classical.brighteners import build_brightener
from lowlight.classical.denoisers import build_denoiser
from lowlight.metrics import resolve_device
from lowlight.models import build_model, count_parameters
from lowlight.pipelines.base import BasePipeline
from lowlight.pipelines.brighten_then_denoise import BrightenThenDenoisePipeline
from lowlight.pipelines.components import (
    ClassicalBrightenerComponent,
    ClassicalDenoiserComponent,
    NeuralComponent,
)
from lowlight.pipelines.denoise_then_brighten import DenoiseThenBrightenPipeline
from lowlight.pipelines.end_to_end import EndToEndPipeline
from lowlight.training.checkpointing import CheckpointManager

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _checkpoint_path(spec: dict[str, object], root: Path, seed: int) -> Path:
    if "checkpoint" in spec:
        return resolve_path(str(spec["checkpoint"]), root)
    root_value = spec.get("checkpoint_root")
    if root_value is None:
        raise ValueError("neural component requires checkpoint or checkpoint_root")
    return resolve_path(str(root_value), root) / f"seed{seed}" / "best.pt"


def _load_neural_component(
    spec: dict[str, object],
    *,
    root: Path,
    seed: int,
    device: str | torch.device,
    name: str,
) -> NeuralComponent:
    checkpoint_path = _checkpoint_path(spec, root, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    resolved_device = resolve_device(None if device == "auto" else str(device))
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model_config = payload.get("config", {}).get("model")
    if model_config is None:
        config_path = spec.get("model_config")
        if config_path is None:
            raise ValueError(f"{checkpoint_path} lacks model config; set model_config")
        model_config = load_yaml(resolve_path(str(config_path), root))["model"]
    model = build_model(dict(model_config)).to(resolved_device)
    CheckpointManager.load(checkpoint_path, model, device=resolved_device)
    model.eval()
    try:
        relative = str(checkpoint_path.relative_to(root))
    except ValueError:
        relative = str(checkpoint_path)
    return NeuralComponent(
        model,
        device=resolved_device,
        name=name,
        checkpoint=relative,
        seed=int(payload.get("seed", seed)),
        epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
        parameters=count_parameters(model),
    )


def build_denoiser_component(
    spec: dict[str, object],
    *,
    root: Path = ROOT,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> ClassicalDenoiserComponent | NeuralComponent:
    kind = str(spec.get("kind", "classical"))
    if kind == "classical":
        if "name" in spec:
            denoiser = build_denoiser(str(spec["name"]), dict(spec.get("params", {})))
            return ClassicalDenoiserComponent(denoiser)
        config_path = spec.get("config")
        if config_path is None:
            raise ValueError("classical denoiser requires config or name")
        payload = load_yaml(resolve_path(str(config_path), root))
        denoiser = build_denoiser(str(payload["name"]), dict(payload["params"]))
        return ClassicalDenoiserComponent(denoiser)
    if kind == "cnn":
        return _load_neural_component(spec, root=root, seed=seed, device=device, name="denoiser")
    raise ValueError(f"unknown denoiser kind: {kind!r}")


def build_brightener_component(
    spec: dict[str, object],
    *,
    root: Path = ROOT,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> ClassicalBrightenerComponent | NeuralComponent:
    kind = str(spec.get("kind", "classical"))
    if kind == "classical":
        if "name" in spec:
            brightener = build_brightener(str(spec["name"]), dict(spec.get("params", {})))
            return ClassicalBrightenerComponent(brightener)
        config_path = spec.get("config")
        if config_path is None:
            raise ValueError("classical brightener requires config or name")
        payload = load_yaml(resolve_path(str(config_path), root))
        track = str(spec.get("track", "raw"))
        method = spec.get("method")
        if method is None:
            method = payload["tracks"][track]["best_method"]
        params = payload["tracks"][track]["methods"][str(method)]["params"]
        brightener = build_brightener(str(method), dict(params))
        return ClassicalBrightenerComponent(brightener)
    if kind == "cnn":
        return _load_neural_component(spec, root=root, seed=seed, device=device, name="brightener")
    raise ValueError(f"unknown brightener kind: {kind!r}")


def _end_to_end_spec(config: dict[str, object], variant: str | None) -> dict[str, object]:
    variants = config.get("variants")
    if variants is None:
        return dict(config["model"])
    if variant is None:
        raise ValueError(f"end-to-end config requires variant: {sorted(variants)}")
    if variant not in variants:
        raise ValueError(f"unknown end-to-end variant: {variant!r}")
    merged = dict(config["model"])
    merged.update(dict(variants[variant]))
    return merged


def build_pipeline(
    config: str | Path | dict[str, object],
    *,
    root: Path = ROOT,
    seed: int = 0,
    device: str | torch.device = "auto",
    variant: str | None = None,
) -> BasePipeline:
    if isinstance(config, dict):
        payload = dict(config)
    else:
        payload = load_yaml(resolve_path(config, root))
    order = str(payload.get("order", ""))
    if order == "denoise_then_brighten":
        denoiser = build_denoiser_component(
            dict(payload["denoiser"]), root=root, seed=seed, device=device
        )
        brightener = build_brightener_component(
            dict(payload["brightener"]), root=root, seed=seed, device=device
        )
        return DenoiseThenBrightenPipeline(denoiser, brightener)
    if order == "brighten_then_denoise":
        brightener = build_brightener_component(
            dict(payload["brightener"]), root=root, seed=seed, device=device
        )
        denoiser = build_denoiser_component(
            dict(payload["denoiser"]), root=root, seed=seed, device=device
        )
        return BrightenThenDenoisePipeline(brightener, denoiser)
    if order == "end_to_end":
        model_spec = _end_to_end_spec(payload, variant)
        component = _load_neural_component(
            model_spec, root=root, seed=seed, device=device, name="end_to_end"
        )
        return EndToEndPipeline(component)
    raise ValueError(f"unknown pipeline order: {order!r}")
