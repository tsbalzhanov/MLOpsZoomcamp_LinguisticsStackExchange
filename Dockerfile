FROM python:3.12.10 AS production

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt update && \
    apt install -y p7zip-full

# build venv

RUN mkdir -p /workspace/build

COPY ./src/pyproject.toml ./src/.python-version ./src/uv.lock /workspace/build

RUN cd /workspace/build && \
    uv sync

# set up workspace

COPY ./src /workspace/src

RUN rm -rf /workspace/src/.venv; \
    mv /workspace/build/.venv /workspace/src && \
    rm -r /workspace/build

WORKDIR /workspace/src
