
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.preprocessing import prepare_data, get_train_test_data
from src.models.evaluate import fit_model, create_eval_table, save_eval_table
from src.ui.sidebar import contract_badge
from setup import logger
from setup.logger import log_call

st.set_page_config(page_title="ML Model Results", layout="wide", page_icon="📊")

if "option_obj" not in st.session_state:
    st.warning("Please first set up an option contract on the home page – no session data found.")
    st.stop()

contract_badge()
opt = st.session_state["option_obj"]

ticker, opt_type = opt.underlying_ticker, opt.option_type

dataset_key = f"{ticker}_{opt_type}"

split_cache  = st.session_state.setdefault("split_cache", {})
eval_cache   = st.session_state.setdefault("eval_cache",  {})
model_cache  = st.session_state.setdefault("model_cache", {})

run_ml = st.button("🚀 Run / Update ML analysis", type="primary")

needs_training = dataset_key not in eval_cache

if needs_training and not run_ml:
    st.info("No ML results for this contract yet. Click the button above to train and evaluate models.")
    st.stop()

if dataset_key not in split_cache or run_ml:   # allow re‑split on force run
    raw_df = prepare_data(ticker=ticker, opt_type=opt_type)
    split_cache[dataset_key] = get_train_test_data(raw_df)
X_train, X_test, y_train, y_test = split_cache[dataset_key]

if needs_training or run_ml:
    with st.spinner("Training & evaluating models – this may take a while …"):
        eval_df = create_eval_table(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        eval_cache[dataset_key] = eval_df
        save_eval_table(eval_df)
        # Reset model cache placeholders
        model_cache[dataset_key] = {m.split("_")[0]: None for m in eval_df["Model_ID"].unique()}
        logger.info("Models trained & cached for %s / %s", ticker, opt_type)
else:
    eval_df = eval_cache[dataset_key]

@log_call(logger)
def predict_price(
    model_name: str,
    *,
    strike: float,
    days_to_maturity: int,
    impliedVolatility: float,
    option_type: str = "call",
) -> float:
    """Return predicted *lastPrice* for a single option spec."""

    mdl_bucket = model_cache.setdefault(dataset_key, {})
    predictor = mdl_bucket.get(model_name)

    if predictor is None:
        predictor, _ = fit_model(
            model_name,
            force=False,
            X_train=X_train,
            y_train=y_train,
        )
        if model_name != "linear":
            predictor = predictor.best_estimator_
        mdl_bucket[model_name] = predictor

    features = pd.DataFrame({
        "strike": [strike],
        "days_to_maturity": [days_to_maturity],
        "impliedVolatility": [impliedVolatility],
        "option_type": [option_type],
    })
    return float(predictor.predict(features)[0])

st.title(f"📊 ML Model Results — {ticker} / {opt_type}")

st.subheader("Performance on the hold‑out test set")
st.dataframe(eval_df, use_container_width=True)

with st.expander("🔍 Compare individual metrics"):
    metric = st.selectbox("Metric", ("MAE", "RMSE", "R2"), index=0)
    st.bar_chart(eval_df.set_index("Model_ID")[metric])

st.divider()
st.header("🧮 Predict a single option price")

c1, c2 = st.columns(2)
with c1:
    chosen_model = st.selectbox(
        "Model",
        options=sorted(model_cache[dataset_key].keys()),
    )
    strike_val = st.number_input(
        "Strike price",
        min_value=0.0,
        value=float(opt.strike_price),
        step=0.5,
    )
    d2m_val = st.number_input(
        "Days to maturity",
        min_value=1,
        value=int(opt.maturity),
        step=1,
    )
with c2:
    iv_val = st.number_input(
        "Implied volatility (e.g. 0.35%)",
        min_value=0.0,
        max_value=5.0,
        value=float(getattr(opt, "volatility", 0.35)),
        step=0.01,
        format="%.2f",
    )
    option_choice = st.selectbox(
        "Option type",
        ["call", "put"],
        index=0 if opt.option_type == "call" else 1,
    )

if st.button("Predict price", type="primary"):
    pred = predict_price(
        chosen_model,
        strike=strike_val,
        days_to_maturity=d2m_val,
        impliedVolatility=iv_val,
        option_type=option_choice,
    )
    st.success(f"Predicted option price: **{pred:.4f}**")
