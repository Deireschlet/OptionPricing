# Python Project Template

Welcome to the **Python Project Template** — your go-to starting point for small Python projects. This template is here to make your life easier (and maybe even a little more fun) by setting up the essentials so you can focus on what really matters: writing great code.

Whether you're working on a side project, experimenting with a new idea, or just want a quick start, this template has your back. It includes a package manager, logging setup, and a configuration loader, all ready to go!

## Features

- **Package Management with uv**: Say goodbye to manually managing your environment. This template uses [**uv**](https://github.com/astral-sh/uv), a super-fast Python package manager that handles dependencies and virtual environments for you. It’s like having a personal assistant for your project.
  
- **Logger Setup**: No more manually setting up loggers every time. With our pre-configured logger, you get neat logs in both the console and a rotating log file. Never miss an important error or warning again!

- **Configuration Loader**: Centralized, clean configuration management! Define your project settings in a simple `config.ini` file and access them from anywhere. It’s like having a settings dashboard for your code, but cooler.

### Password Encryption Script – How to Use

This script helps you securely store an **encrypted password or phrase** in a `.env` file using a generated encryption key (`pass.key`). It keeps sensitive data out of version control while still allowing secure access in your project.

#### How to use it:

1. **Run the script**  
   This will:
   - Prompt you for a password
   - Create `pass.key` (if it doesn't exist)
   - Encrypt the password
   - Save it to a `.env` file under `ENCRYPTED_PASSWORD`

2. **Use the password in your code**  
   Import the required functions and load the `.env`:
   ```python
   from dotenv import load_dotenv
   from setup.utils import decrypt_password, load_key
   import os

   load_dotenv()
   encrypted_pw = os.getenv("ENCRYPTED_PASSWORD")
   password = decrypt_password(encrypted_pw, load_key())

⚠️ This project is a work in progress — things might change, break, or magically improve at any time!
