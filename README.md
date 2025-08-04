# Common env variables
```shell
STACK_EXCHANGE_COMMUNITY=linguistics
ROOT_DATA_DIR=work_data
RANDOM_SEED=63
MLFLOW_DATA_DIR="$ROOT_DATA_DIR/mlflow"
```

# Launch mlflow tracking server
```shell
mkdir -p $MLFLOW_DATA_DIR
uv run mlflow server --backend-store-uri "sqlite:///$MLFLOW_DATA_DIR/store.db" --default-artifact-root $MLFLOW_DATA_DIR/artifacts
```

# Launch prefect orchestrator & runner servers
```shell
uv run prefect server start
PREFECT_API_URL=http://127.0.0.1:4200/api uv run python main.py
```

# Launch workflows
```shell
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_NAME=sxl
MODEL_ALIAS=prod

uv run prefect deployment run "download-and-unpack-archive/download-and-unpack-archive" -p root_data_dir="$ROOT_DATA_DIR" -p community=$STACK_EXCHANGE_COMMUNITY

uv run prefect deployment run "prepare-dataset/prepare-dataset" -p root_data_dir="$ROOT_DATA_DIR" -p random_seed=$RANDOM_SEED -p validation_size=0.1 -p test_size=0.3

uv run prefect deployment run "train-model/train-model" -p root_data_dir="$ROOT_DATA_DIR" -p random_seed=$RANDOM_SEED -p mlflow_tracking_uri=$MLFLOW_TRACKING_URI -p model_name=$MODEL_NAME -p model_alias=$MODEL_ALIAS

uv run prefect deployment run "prepare-serving-config/prepare-serving-config" -p root_data_dir="$ROOT_DATA_DIR" -p mlflow_tracking_uri=$MLFLOW_TRACKING_URI -p model_name=$MODEL_NAME -p model_alias=$MODEL_ALIAS
```

# Serve model via mlserver

```shell
uv run mlserver start work_data/mlserver
```

# Run linter, isort & type checker

```shell
uv run ruff check && uv run isort --check --diff . && uv run mypy .
```
