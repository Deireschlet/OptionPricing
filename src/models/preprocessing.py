
# load data and create preprocessor for pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.data_handler import fetch_option_data
from setup import logger, config
from setup.logger import log_call


@log_call(logger)
def prepare_data(ticker: str=None, opt_type: str=None):
    if ticker and opt_type:
        df = fetch_option_data(ticker=ticker, opt_type=opt_type)
        df = df.assign(option_type=opt_type).reset_index()
        return df
    else:
        return None


@log_call(logger)
def create_preprocessor():
    numeric_features = ['strike', 'days_to_maturity', 'impliedVolatility']
    category_features = ['option_type']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(with_mean=True, with_std=True), numeric_features),
        ('cat', OneHotEncoder(), category_features)
    ])

    return preprocessor


@log_call(logger)
def get_train_test_data(df: pd.DataFrame=None):
    numeric_features = ['strike', 'days_to_maturity', 'impliedVolatility']
    category_features = ['option_type']

    try:
        X = df[numeric_features + category_features]
        y = df['lastPrice']
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(e)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass