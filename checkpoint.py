"""Checkpoint I/O helpers with fp16/fp32 compatibility."""

from __future__ import annotations

import torch


META_KEYS = ("net", "height", "width", "dinov2_encoder", "_weight_dtype")


def load_checkpoint(path, map_location=None):
    """Load a DA360 checkpoint (fp32 or fp16 tensors)."""
    return torch.load(path, map_location=map_location)


def checkpoint_meta(ckpt, defaults=None):
    """Return model construction metadata with defaults filled in."""
    defaults = defaults or {}
    return {
        "net": ckpt.get("net", defaults.get("net", "DA360")),
        "dinov2_encoder": ckpt.get(
            "dinov2_encoder", defaults.get("dinov2_encoder", "vits")
        ),
        "height": ckpt.get("height", defaults.get("height", 518)),
        "width": ckpt.get("width", defaults.get("width", 1036)),
    }


def _alias_key(key):
    """Map ERPCircularConv2d alias <-> real conv parameter names."""
    if key.endswith(".conv.weight"):
        return key[: -len(".conv.weight")] + ".weight"
    if key.endswith(".conv.bias"):
        return key[: -len(".conv.bias")] + ".bias"
    if key.endswith(".weight") and not key.endswith(".conv.weight"):
        return key[: -len(".weight")] + ".conv.weight"
    if key.endswith(".bias") and not key.endswith(".conv.bias"):
        return key[: -len(".bias")] + ".conv.bias"
    return None


def strip_redundant_checkpoint_keys(ckpt):
    """Drop duplicated ERPCircularConv aliases and BN step counters.

    ``ERPCircularConv2d`` used to register the same Parameter under both
    ``*.weight`` and ``*.conv.weight`` (and likewise for bias). Saving the
    flattened state_dict then stored each tensor twice (~5MB in fp16).
    Keep the real ``*.conv.*`` entries; drop the aliases.
    """
    drop = set()
    for key, value in ckpt.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith("num_batches_tracked"):
            drop.add(key)
            continue
        if key.endswith(".weight") and not key.endswith(".conv.weight"):
            alt = key[: -len(".weight")] + ".conv.weight"
            if alt in ckpt and torch.is_tensor(ckpt[alt]):
                drop.add(key)
        elif key.endswith(".bias") and not key.endswith(".conv.bias"):
            alt = key[: -len(".bias")] + ".conv.bias"
            if alt in ckpt and torch.is_tensor(ckpt[alt]):
                drop.add(key)
    return {k: v for k, v in ckpt.items() if k not in drop}


def state_dict_from_checkpoint(ckpt, model_state_dict):
    """Filter checkpoint tensors to match ``model`` and cast dtypes if needed.

    fp16 checkpoints are cast to the parameter dtype (typically fp32) so
    ``load_state_dict`` works for both full-precision and half-precision files.
    Also resolves ``*.weight`` <-> ``*.conv.weight`` aliases from older dumps.
    """
    out = {}
    for key, target in model_state_dict.items():
        value = ckpt.get(key)
        if value is None or not torch.is_tensor(value):
            alt = _alias_key(key)
            if alt is not None:
                value = ckpt.get(alt)
        if value is None or not torch.is_tensor(value):
            continue
        if value.dtype != target.dtype:
            value = value.to(dtype=target.dtype)
        if value.device != target.device:
            value = value.to(device=target.device)
        out[key] = value
    return out
