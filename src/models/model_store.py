import pandas as pd
from typing import Any
import joblib
from pathlib import Path
import hashlib

from setup import logger, config
from setup.logger import log_call


_MODEL_DIR = config.get("PROJECT", "models_path")
_MODEL_DIR.mkdir(exist_ok=True)


@log_call(logger)
def _data_hash(arrays) -> str:
    """Return a short SHA-256 hash of the concatenated raw bytes."""
    m = hashlib.sha256()
    for arr in arrays:
        m.update(
            arr.to_numpy().tobytes()
            if isinstance(arr, (pd.Series, pd.DataFrame))
            else arr.tobytes()
        )
    return m.hexdigest()[:12]          # 12-char prefix is fine


@log_call(logger)
def _model_path(model_name: str, data_hash: str) -> Path:
    return _MODEL_DIR / f"{model_name}_{data_hash}.joblib"


@log_call(logger)
def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, path, compress=("xz", 3))


@log_call(logger)
def _load_model(path: Path) -> Any | None:
    return joblib.load(path) if path.exists() else None


if __name__ == "__main__":
    pass
