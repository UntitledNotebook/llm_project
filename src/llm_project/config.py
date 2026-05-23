from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Small dot-access wrapper around a nested dict.

    The training scripts intentionally avoid large framework abstractions so that the core SFT
    and GRPO logic is visible. This class is only for ergonomic config access.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix}")
    if data is None:
        data = {}
    return _wrap(data)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain_dict(config), f, sort_keys=False)


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)  # type: ignore[index]
        else:
            out[key] = copy.deepcopy(value)
    return out
