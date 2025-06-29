import datetime as dt

from setup import logger
from setup.logger import log_call


START_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
END_DATE: str = dt.date.today().strftime("%Y-%m-%d")


@log_call(logger)
def main():
    pass


if __name__ == "__main__":
    main()
