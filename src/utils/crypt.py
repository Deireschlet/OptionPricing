from getpass import getpass
from cryptography.fernet import Fernet
from pathlib import Path
from dotenv import set_key, load_dotenv
from setup import logger

KEY_FILE = Path(__file__).resolve().parents[0] / "pass.key"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def generate_key() -> bytes:
    return Fernet.generate_key()


def save_key(key: bytes):
    with open(KEY_FILE, "wb") as f:
        f.write(key)


def load_key() -> bytes:
    with open(KEY_FILE, "rb") as f:
        return f.read()


def encrypt_password(password: str, key: bytes) -> str:
    fernet = Fernet(key)
    return fernet.encrypt(password.encode()).decode()


def write_to_env(encrypted_pw: str):
    # Create .env if it doesn't exist
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    # Load and update .env
    load_dotenv(dotenv_path=ENV_FILE)
    set_key(str(ENV_FILE), "ENCRYPTED_PASSWORD", encrypted_pw)


def decrypt_password(encrypted_pw: str, key: bytes) -> str:
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_pw.encode()).decode()


def main():
    if not KEY_FILE.exists():
        key = generate_key()
        save_key(key)
        logger.info(f"Key generated and saved to {KEY_FILE}\n")
    else:
        key = load_key()
        logger.info(f"Using existing key from {KEY_FILE}\n")

    password = getpass("Enter password to encrypt: ")
    encrypted_pw = encrypt_password(password, key)
    write_to_env(encrypted_pw)
    logger.info(f"Encrypted password written to {ENV_FILE} as ENCRYPTED_PASSWORD")


if __name__ == "__main__":
    main()
