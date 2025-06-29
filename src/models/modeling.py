# creates pipelines for training
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor

from .preprocessing import create_preprocessor
from setup import logger
from setup.logger import log_call


@log_call(logger)
def get_pipeline(model: str=None):
    if model == "linear":
        linreg_pipeline = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', LinearRegression())
        ])
        return linreg_pipeline
    elif model == "randomforest":
        rf_pipeline = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('regressor', RandomForestRegressor())
        ])
        return rf_pipeline
    elif model == "xgboost":
        xgb_pipeline = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('regressor', XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",  # or "gpu_hist" if you have a GPU
                random_state=42,
                n_jobs=-1  # use all CPU cores
            ))
        ])
        return xgb_pipeline
    else:
        return None


@log_call(logger)
def get_param_grid(model: str=None):
    if model == "linear":
        return {
            'regressor__fit_intercept': [True, False],
            'regressor__normalize': [True, False]
        }
    elif model == "randomforest":
        return {
    "regressor__n_estimators":     [200, 800],          # more trees ⇒ lower variance
    "regressor__max_depth":       [10, 20],         # None = grow fully
    "regressor__min_samples_split":[5, 10],               # node-split granularity
    "regressor__min_samples_leaf":[2, 4],                 # leaf size regularisation
    "regressor__max_features":    ["sqrt", 0.8]      # feature subsampling
}
    elif model == "xgboost":
        return {
    # ensemble size / learning dynamics
    "regressor__n_estimators" : randint(300, 1200),
    "regressor__learning_rate" : uniform(0.01, 0.19),
    # tree complexity
    "regressor__max_depth" : randint(3, 9),
    "regressor__min_child_weight" : uniform(0.5, 9.5),
    # regularisation
    "regressor__subsample" : uniform(0.5, 1.0),
    "regressor__colsample_bytree" : uniform(0.5, 1.0),
    "regressor__gamma" : uniform(0, 5),
    "regressor__reg_alpha" : uniform(0, 1),
    "regressor__reg_lambda" : uniform(1, 4),
}
    else:
        return None


@log_call(logger)
def get_model(model: str=None, num_splits: int=5):
    tscv = TimeSeriesSplit(n_splits=num_splits)
    if model == "randomforest":
        rf_grid = GridSearchCV(
            estimator=get_pipeline(model=model),
            param_grid=get_param_grid(model=model),
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        return rf_grid
    elif model == "xgboost":
        xgb_search = RandomizedSearchCV(
            estimator=get_pipeline(model=model),
            param_distributions=get_param_grid(model=model),
            n_iter=64,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=0,
            n_jobs=-3,
            refit=True,
            random_state=42
        )
        return xgb_search
    elif model == "linear":
        return get_pipeline(model=model)
    else:
        return None


if __name__ == "__main__":
    pass