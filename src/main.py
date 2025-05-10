from setup import config
from setup import logger
from setup.logger import log_call


@log_call(logger)
def main():
    print(f"Hello {config.get("PROJECT", "name")}")


if __name__ == "__main__":
    main()
