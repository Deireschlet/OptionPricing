from setup import config

LINEAR_MODEL = config.get("MODELS", "linear_regression")
RANDOM_FOREST_MODEL = config.get("MODELS", "random_forest")
XGB_MODEL = config.get("MODELS", "xgboost")

MODELS = [LINEAR_MODEL, RANDOM_FOREST_MODEL, XGB_MODEL]