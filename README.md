# MLOps project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

This is a final project for MLOps Zoomcamp 2025. It uses open data from Data Science Stack Exchange. The data for this and other stack exchange / stack overflow sites can be downloaded from archive.org.

## Problem Description

The project is built around a classification model which would predict whether a topic from a stack exchange forum has been answered depending on several numerical features number of votes, title length (in number of symbols), number of comments, whether post author has description etc.

## Technologies

The project uses following technologies:

- Docker for containerization
- Docker Compose for managing containers
- Prefect for flow orcherstration with PostgreSQL as a DBMS for Prefect
- LightGBM as a ML framework
- MLflow for model tracking & model registry
- MLServer as an inference server
- uv & venv for managing python environment
- ruff, isort & mypy for statically ensuring code quality and enforcing codestyle

## Workflow description

There are 4 workflows in the pipeline:

1. `download_and_unpack_archive` - downloads archive with Stack Exchange forum posts from archive.org and unpack its
2. `prepare_dataset` - prepares dataset from initial archive: builds features and performs split into train, validation & test
3. `train_model` - trains LightGBM model from prepared dataset
4. `prepare_mlserver_resources` - prepares resources from previously trained model for serving with MLServer

## How to reproduce

Requirements:
- Docker + Docker Compose
- uv + python

0. (optionally) Statically validate code
In `src` directory:

```shell
uv run ruff check && uv run isort --check --diff . && uv run mypy
```

1. Build docker images and setup docker containers:
```shell
docker compose build && \
docker compose up
```

2. Visit Prefect UI dashboard on `http://localhost:18900` and launch workflows in order. Examples of parameters for workflows:
    1. `download_and_unpack_archive`: `community` = `linguistics`
    2. `prepare_dataset`: `validation_size` = `0.1`, `test_size` = `0.3`
    3. `train_model`: `model_name` = `linguistics_lgbm`, `model_alias` = `prod`
    4. `prepare_mlserver_resources`: same as for `train_model`


3. After that, model would be available for inference via MLServer on `http://localhost:18910` (restart might be required)
