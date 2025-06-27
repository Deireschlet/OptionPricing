import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .modeling import get_model
from setup import logger, config
from setup.logger import log_call

from src.models import MODELS
from .model_store import _data_hash, _load_model, _save_model, _model_path


@log_call(logger)
def fit_model(model_name: str, *, force: bool=False, X_train=None, y_train=None):
    """
    Train or load a model.  Returns (fitted_model, model_id).
    model_id = '<model_name>_<12-char-hash>'.
    """

    if model_name is None:
        return None, None

    data_h = _data_hash([X_train, y_train])
    model_id = f"{model_name}_{data_h}"
    path = _model_path(model_name, data_h)

    if not force:
        cached = _load_model(path)
        if cached is not None:
            logger.info(f"Loaded {path.name}")
            return cached, model_id

    mdl = get_model(model_name)
    mdl.fit(X_train, y_train)
    _save_model(mdl, path)
    return mdl, model_id


@log_call(logger)
def evaluate_model(fitted_model, model_name, X_test=None, y_test=None):
    y_pred = fitted_model.predict(X_test) if model_name == "linear" else fitted_model.best_estimator_.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2


@log_call(logger)
def create_eval_table(models: list[str] | None=MODELS, X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Build a DataFrame with columns [Model_ID, MAE, RMSE, R2].
    Model_ID contains the data-hash suffix so you can see *exactly*
    which model / data combo each row refers to.
    """
    if not models:
        return pd.DataFrame(columns=["Model_ID", "MAE", "RMSE", "R2"])

    rows = []
    for model_name in models:
        logger.info(f"Start fitting model {model_name} ...")
        fitted, model_id = fit_model(model_name, X_train=X_train, y_train=y_train)
        logger.info(f"Start evaluating model {model_name} ...")# now returns id
        mae, rmse, r2 = evaluate_model(fitted, model_name, y_test=y_test, X_test=X_test)
        logger.info(f"Finished evaluating model {model_name}")
        rows.append([model_id, mae, rmse, r2])

    return pd.DataFrame(rows, columns=["Model_ID", "MAE", "RMSE", "R2"])


@log_call(logger)
def save_eval_table(
    df: pd.DataFrame,
    path: str | Path = config.get("PROJECT", "eval_table_path")
) -> None:

    clean_path = str(path).strip(" \"'")
    p = Path(clean_path).expanduser().resolve()

    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p, index=False)
    logger.info("Saved evaluation table to %s", p)


@log_call(logger)
def predict_price(
    model_name: str,
    *,
    strike: float,
    days_to_maturity: int,
    impliedVolatility: float,
    option_type: str = "call",
) -> float:
    """Return the *lastPrice* prediction for the given option characteristics.

    Parameters
    ----------
    model_name
        One of the model identifiers you used during training ("linear", "randomforest", "xgboost").
    strike
        Option strike price.
    days_to_maturity
        Calendar days until the option expires.
    impliedVolatility
        Black‑Scholes‑style implied volatility (e.g. ``0.32`` for 32 %).
    option_type
        Either ``"call"`` or ``"put"``. Must match the values seen during training.

    Returns
    -------
    float
        Predicted price (same units as the *lastPrice* column in the training data).
    """

    # Load the chosen model
    mdl, _ = fit_model(model_name, force=False)

    features = pd.DataFrame(
        {
            "strike": [strike],
            "days_to_maturity": [days_to_maturity],
            "impliedVolatility": [impliedVolatility],
            "option_type": [option_type],
        }
    )

    predictor = mdl if model_name == "linear" else mdl.best_estimator_
    return float(predictor.predict(features)[0])


if __name__ == "__main__":
    result = create_eval_table()
    save_eval_table(result)

