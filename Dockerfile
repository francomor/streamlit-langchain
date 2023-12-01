FROM python:3.11-slim-buster as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.6.1

RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes --with-credentials  \
    | /venv/bin/pip install -r /dev/stdin

FROM base as final

COPY --from=builder /venv /venv
COPY . .

ENV SERVER_NAME="0.0.0.0"
ENV SERVER_PORT="8080"
ENV STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
RUN chmod +x entry_point.sh
CMD ["./entry_point.sh"]
