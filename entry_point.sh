#!/usr/bin/env bash

set -e
. /venv/bin/activate

cd /app
streamlit run home.py --server.port=$SERVER_PORT --server.address=$SERVER_NAME
