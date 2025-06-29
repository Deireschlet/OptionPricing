import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .modeling import get_model
from setup import logger, config
from setup.logger import log_call
from typing import Optional

from src.models import MODELS
from .model_store import _data_hash, _save_model, _model_path, get_latest_model


@log_call(logger)
def fit_model(model_name: str, *, ticker: str, force: bool = False, X_train=None, y_train=None,
              option_type: Optional[str] = None):
    """
    Train or load a model.

    Parameters
    ----------
    model_name : str
        Name of the model to train or load.
    ticker : str
        Ticker symbol for which the model is being trained.
    force : bool, optional
        If True, force retrain even if a cached model exists.
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    option_type : str, optional
        Type of option ('call' or 'put') for model specialization.

    Returns
    -------
    tuple
        (fitted_model, model_id)
    """
    if model_name is None:
        return None, None

    # If not forcing a retrain, try to get the latest model
    if not force and ticker and option_type:
        model, path = get_latest_model(model_name, ticker, option_type)
        if model is not None:
            model_id = path.stem
            logger.info(f"Using latest {ticker} {option_type} model: {model_id}")
            return model, model_id

    # Generate data hash for this training set
    data_h = _data_hash([X_train, y_train])

    # Create a model path that includes ticker and option type
    path = _model_path(model_name, ticker, data_h, option_type)
    model_id = path.stem

    # Train the model
    mdl = get_model(model_name)
    mdl.fit(X_train, y_train)
    _save_model(mdl, path)

    return mdl, model_id


@log_call(logger)
def evaluate_model(fitted_model, model_name, X_test=None, y_test=None):
    """Evaluate a model on test data and return metrics."""
    if model_name == "linear":
        y_pred = fitted_model.predict(X_test)
    else:
        if hasattr(fitted_model, 'best_estimator_'):
            y_pred = fitted_model.best_estimator_.predict(X_test)
        else:
            y_pred = fitted_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


@log_call(logger)
def create_eval_table(models: list[str] | None = MODELS, ticker: str = None, X_train=None, y_train=None, X_test=None,
                      y_test=None, option_type=None):
    """
    Build a DataFrame with columns [Model_ID, MAE, RMSE, R2].
    Model_ID contains the model name, ticker, option type, and a timestamp.
    """
    if not models:
        return pd.DataFrame(columns=["Model_ID", "MAE", "RMSE", "R2"])

    # Extract ticker if not provided
    if ticker is None:
        logger.warning("No ticker provided for create_eval_table. Models may not be correctly stored.")
        ticker = "unknown"

    # Extract option_type from the data if not provided
    if option_type is None and X_train is not None and "option_type" in X_train.columns:
        unique_types = X_train["option_type"].unique()
        if len(unique_types) == 1:
            option_type = unique_types[0]
            logger.info(f"Detected option_type={option_type} from training data")
        else:
            logger.warning(f"Multiple option types detected in data: {unique_types}. Models won't be specialized.")

    rows = []
    for model_name in models:
        logger.info(f"Start fitting model {model_name} for ticker={ticker}, option_type={option_type} ...")
        fitted, model_id = fit_model(
            model_name,
            ticker=ticker,
            X_train=X_train,
            y_train=y_train,
            option_type=option_type
        )
        logger.info(f"Start evaluating model {model_name} ...")
        mae, rmse, r2 = evaluate_model(fitted, model_name, y_test=y_test, X_test=X_test)
        logger.info(f"Finished evaluating model {model_name}")

        # Store just the model name in the model_id for display purposes
        simple_model_id = model_name
        rows.append([simple_model_id, mae, rmse, r2])

    return pd.DataFrame(rows, columns=["Model_ID", "MAE", "RMSE", "R2"])


@log_call(logger)
def save_eval_table(
        df: pd.DataFrame,
        path: str | Path = config.get("PROJECT", "eval_table_path")
) -> None:
    """Save the evaluation table to a CSV file."""
    clean_path = str(path).strip(" \"'")
    p = Path(clean_path).expanduser().resolve()

    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p, index=False)
    logger.info("Saved evaluation table to %s", p)


@log_call(logger)
def predict_price(
        model_name: str,
        ticker: str,
        *,
        strike: float,
        days_to_maturity: int,
        impliedVolatility: float,
        option_type: str = "call",
) -> float:
    """
    Return the *lastPrice* prediction for the given option characteristics.
    Uses the latest trained model for the specified ticker, model_name and option_type.
    """
    # Load the latest model for this ticker, model_name and option_type
    model, _ = get_latest_model(model_name, ticker, option_type)

    if model is None:
        # Fall back to any model of this ticker and type if no option-specific model exists
        model, _ = get_latest_model(model_name, ticker)

    if model is None:
        raise ValueError(f"No trained model found for ticker={ticker}, model={model_name}, option_type={option_type}")

    features = pd.DataFrame(
        {
            "strike": [strike],
            "days_to_maturity": [days_to_maturity],
            "impliedVolatility": [impliedVolatility],
            "option_type": [option_type],
        }
    )

    # Handle different model types correctly
    if model_name == "linear":
        predictor = model
    else:
        if hasattr(model, 'best_estimator_'):
            predictor = model.best_estimator_
        else:
            predictor = model

    try:
        return float(predictor.predict(features)[0])
    except ValueError as e:
        if "Found unknown categories" in str(e):
            raise ValueError(
                f"Model was not trained with option_type='{option_type}'. Please use a model trained with this option type.")
        raise


if __name__ == "__main__":
    result = create_eval_table()
    save_eval_table(result)