"""Configuration loading and path resolution.

Every path in DemandShock is resolved through this module with pathlib, relative
to the repository root. No machine-specific path is ever hardcoded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, unreadable, or internally invalid."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    # ---- generic access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # ---- mode -----------------------------------------------------------
    @property
    def mode(self) -> str:
        return str(self.raw["mode"])

    @property
    def is_dev(self) -> bool:
        return self.mode == "development"

    @property
    def mode_settings(self) -> dict[str, Any]:
        return self.raw["development" if self.is_dev else "full"]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    # ---- paths ----------------------------------------------------------
    def path_of(self, key: str) -> Path:
        return (REPO_ROOT / self.raw["paths"][key]).resolve()

    @property
    def m5_dir(self) -> Path:
        return self.path_of("m5_dir")

    @property
    def external_dir(self) -> Path:
        return self.path_of("external_dir")

    @property
    def processed_dir(self) -> Path:
        return self.path_of("processed_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self.path_of("artifacts_dir")

    @property
    def models_dir(self) -> Path:
        return self.path_of("models_dir")

    @property
    def logs_dir(self) -> Path:
        return self.path_of("logs_dir")

    def m5_file(self, key: str) -> Path:
        return self.m5_dir / self.raw["files"][key]

    def external_file(self, key: str) -> Path:
        return self.external_dir / self.raw["files"][key]

    def fred_file(self, state: str) -> Path:
        return self.external_dir / self.raw["files"]["fred"][state]

    def ensure_dirs(self) -> None:
        for key in ("processed_dir", "artifacts_dir", "models_dir", "logs_dir"):
            self.path_of(key).mkdir(parents=True, exist_ok=True)

    # ---- lightgbm params ------------------------------------------------
    def lgbm_params(self) -> dict[str, Any]:
        params = dict(self.raw["lgbm"])
        overrides = params.pop("dev_overrides", {}) or {}
        for key in ("num_boost_round", "early_stopping_rounds", "early_stopping_days"):
            params.pop(key, None)
        if self.is_dev:
            params.update(overrides)
        params["seed"] = self.seed
        params["deterministic"] = True
        params["force_row_wise"] = True
        return params

    @property
    def num_boost_round(self) -> int:
        return int(self.raw["lgbm"]["num_boost_round"])

    @property
    def early_stopping_rounds(self) -> int:
        return int(self.raw["lgbm"]["early_stopping_rounds"])

    @property
    def early_stopping_days(self) -> int:
        return int(self.raw["lgbm"]["early_stopping_days"])

    # ---- folds ----------------------------------------------------------
    @property
    def folds(self) -> dict[str, dict[str, int]]:
        return self.raw["folds"]

    def fold(self, name: str) -> dict[str, int]:
        try:
            return self.raw["folds"][name]
        except KeyError as exc:
            raise ConfigError(f"Unknown fold '{name}'") from exc

    @property
    def eval_folds(self) -> list[str]:
        """Folds that have observable actuals (excludes FORWARD)."""
        return [f for f in self.folds if f != "FORWARD"]

    @property
    def cv_folds(self) -> list[str]:
        """Folds used for ablation / residual pooling (excludes HOLDOUT, FORWARD)."""
        return [f for f in self.folds if f not in ("HOLDOUT", "FORWARD")]

    # ---- horizon buckets -------------------------------------------------
    @property
    def horizon_buckets(self) -> list[dict[str, Any]]:
        """Step ranges served by their own model, each with its own origin shift.

        Falls back to a single 28-step bucket so configs predating bucketing keep
        working unchanged.
        """
        buckets = self.raw.get("horizon_buckets")
        if not buckets:
            return [{"name": "H1", "min_step": 1, "max_step": int(self.raw["horizon"]),
                     "shift": int(self.raw["features"]["demand_shift"])}]
        return [dict(b) for b in buckets]

    @property
    def bucket_shifts(self) -> list[int]:
        return sorted({int(b["shift"]) for b in self.horizon_buckets})

    def bucket_for_step(self, step: int) -> dict[str, Any]:
        for bucket in self.horizon_buckets:
            if int(bucket["min_step"]) <= step <= int(bucket["max_step"]):
                return bucket
        raise ConfigError(f"step {step} falls outside every horizon bucket")

    @property
    def train_window_days(self) -> int | None:
        value = self.mode_settings.get("train_window_days")
        return None if value is None else int(value)

    # ---- provenance -----------------------------------------------------
    def hash(self) -> str:
        """Stable hash of the config, recorded in model metadata."""
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


_REQUIRED_TOP_LEVEL = (
    "project", "mode", "paths", "files", "development", "full", "folds",
    "features", "fema", "fred", "lgbm", "ablation", "uncertainty", "shock",
    "inventory", "horizons",
)


def load_config(path: str | Path | None = None, mode: str | None = None) -> Config:
    """Load and validate config.yaml. `mode` overrides the file's mode when given."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(
            f"Config file not found: {cfg_path}\n"
            "Run from the repository root, or pass --config."
        )
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must contain a mapping at the top level.")

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ConfigError(f"config.yaml is missing required keys: {', '.join(missing)}")

    if mode:
        raw = {**raw, "mode": mode}
    if raw["mode"] not in ("development", "full"):
        raise ConfigError(
            f"mode must be 'development' or 'full', got {raw['mode']!r}"
        )

    for name, spec in raw["folds"].items():
        for key in ("train_end_d", "val_start_d", "val_end_d"):
            if key not in spec:
                raise ConfigError(f"fold {name} is missing '{key}'")
        if spec["val_start_d"] <= spec["train_end_d"]:
            raise ConfigError(
                f"fold {name}: val_start_d must be after train_end_d "
                "(chronological validation only)"
            )
        if spec["val_end_d"] < spec["val_start_d"]:
            raise ConfigError(f"fold {name}: val_end_d precedes val_start_d")

    weights = raw["shock"]["weights"]
    total = sum(weights.values())
    if abs(total - 100) > 1e-6:
        raise ConfigError(f"shock.weights must sum to 100, got {total}")

    return Config(raw=raw, path=cfg_path)


def get_logger(name: str, cfg: Config | None = None, filename: str | None = None):
    """Console + optional file logger. Idempotent across repeated calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if cfg is not None and filename:
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.logs_dir / filename, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
