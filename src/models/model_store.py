import pandas as pd
from typing import Any, Optional, Tuple, Dict, List
import joblib
from pathlib import Path
import hashlib
import glob
import os
from datetime import datetime

from setup import logger, config
from setup.logger import log_call

# Model directory setup
_MODEL_DIR = Path(config.get("PROJECT", "models_path"))
_MODEL_DIR.mkdir(exist_ok=True, parents=True)


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
    return m.hexdigest()[:12]  # 12-char prefix is fine


@log_call(logger)
def _model_path(model_name: str, ticker: str, data_hash: str, option_type: Optional[str] = None) -> Path:
    """Generate a path for saving a model with ticker, option type, and timestamp info."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if option_type:
        return _MODEL_DIR / f"{model_name}_{ticker}_{option_type}_{data_hash}_{timestamp}.joblib"
    else:
        return _MODEL_DIR / f"{model_name}_{ticker}_{data_hash}_{timestamp}.joblib"


@log_call(logger)
def _save_model(model: Any, path: Path) -> None:
    """Save a model to the specified path."""
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, path, compress=("xz", 3))
    logger.info(f"Saved model to {path}")


@log_call(logger)
def _load_model(path: Path) -> Any | None:
    """Load a model from the specified path."""
    if path.exists():
        logger.info(f"Loading model from {path}")
        return joblib.load(path)
    return None


@log_call(logger)
def get_latest_model(model_name: str, ticker: str, option_type: Optional[str] = None) -> Tuple[Any | None, Path | None]:
    """
    Get the latest model of the specified type, ticker, and option type.
    Returns the model and its path.
    """
    pattern = f"{model_name}_{ticker}_"
    if option_type:
        pattern += f"{option_type}_"
    pattern += "*.joblib"

    # Find all matching model files
    model_files = list(_MODEL_DIR.glob(pattern))

    if not model_files:
        logger.warning(f"No models found matching pattern: {pattern}")
        return None, None

    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model_path = model_files[0]

    # Load the latest model
    model = _load_model(latest_model_path)
    logger.info(f"Loaded latest model: {latest_model_path.name}")

    return model, latest_model_path


@log_call(logger)
def list_available_models(ticker: Optional[str] = None) -> Dict[str, List[Tuple[str, str, datetime, str]]]:
    """
    List all available models grouped by model name.
    Returns a dictionary with model names as keys and lists of (ticker, option_type, timestamp, path) tuples as values.

    If ticker is provided, only return models for that ticker.
    """
    if ticker:
        pattern = f"*_{ticker}_*.joblib"
    else:
        pattern = "*.joblib"

    model_files = list(_MODEL_DIR.glob(pattern))
    model_info = {}

    for model_path in model_files:
        name_parts = model_path.stem.split('_')
        if len(name_parts) >= 4:  # Should have at least model_name, ticker, option_type/data_hash, timestamp
            model_name = name_parts[0]
            file_ticker = name_parts[1]

            # Check if the third part is an option type
            option_type = name_parts[2] if name_parts[2] in ["call", "put"] else "unknown"

            # Get modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))

            if model_name not in model_info:
                model_info[model_name] = []

            model_info[model_name].append((file_ticker, option_type, mod_time, str(model_path)))

    # Sort each model's entries by timestamp (newest first)
    for model_name in model_info:
        model_info[model_name].sort(key=lambda x: x[2], reverse=True)

    return model_info


@log_call(logger)
def get_model_names_for_ticker(ticker: str) -> List[str]:
    """
    Get a list of unique model names (e.g., 'linear', 'randomforest', 'xgboost')
    that have been trained for a specific ticker.
    """
    models_info = list_available_models(ticker)
    return sorted(list(models_info.keys()))


@log_call(logger)
def clean_old_models(ticker: Optional[str] = None, keep_latest: int = 3) -> None:
    """
    Clean up old models, keeping only the specified number of latest versions for each model type.

    If ticker is specified, only clean models for that ticker.
    """
    model_info = list_available_models(ticker)

    for model_name, models in model_info.items():
        # Group by ticker and option type
        groups = {}
        for file_ticker, opt_type, timestamp, path in models:
            key = (file_ticker, opt_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(path)

        # For each group, keep only the latest models
        for (file_ticker, opt_type), paths in groups.items():
            if len(paths) > keep_latest:
                for path in paths[keep_latest:]:
                    try:
                        Path(path).unlink()
                        logger.info(f"Removed old model: {path}")
                    except Exception as e:
                        logger.error(f"Failed to remove model {path}: {e}")