# Data summary with GPT

Takes a text and summarizes it using GPT.

It uses:
* streamlit
* langchain

The app store the user input in a .pkl file, so it can be used in future executions.
## Installation

```bash
poetry install
```

## Usage

```bash
poetry shell
streamlit run home.py
```

## Credentials

Default user: `admin`

Default password: `admin`

To change the credentials, check the official docs of Streamlit Authenticator: https://github.com/mkhorasani/Streamlit-Authenticator
