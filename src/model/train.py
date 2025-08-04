import logging
import pathlib

import lightgbm
import mlflow
import mlflow.entities
import mlflow.entities.model_registry
import mlflow.lightgbm
import numpy as np
import pandas as pd
import prefect
import sklearn.metrics as sk_metrics  # type: ignore[import-untyped]

from common import (
    TARGET_COLUMN, DatasetSplit, StackExchangeDataset,
    StackExchangePool, get_dataset_dir, get_random_seed, set_up_logger
)


logger = logging.getLogger(__name__)


def _calculate_binary_classification_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == y_pred.ndim == 1
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_pred)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0))
    max_f1_score = float(np.max(f1_scores))
    return {
        'test_roc_auc_score': float(sk_metrics.roc_auc_score(y_true, y_pred)),
        'test_pr_auc_score': float(sk_metrics.auc(recall, precision)),
        'test_max_f1_score': max_f1_score
    }


@prefect.task
def load_dataset(dataset_dir: pathlib.Path) -> StackExchangeDataset:
    return {
        split: StackExchangePool(
            pd.read_parquet(dataset_dir / f'{split}_features.parquet'),
            pd.read_parquet(dataset_dir / f'{split}_target.parquet')[TARGET_COLUMN]
        )
        for split in DatasetSplit
    }


@prefect.task
def train_model_inner(dataset: StackExchangeDataset) -> mlflow.entities.Run:
    logger.info('Training LightGBM model')
    random_seed = get_random_seed()
    mlflow.lightgbm.autolog()
    train_pool = lightgbm.Dataset(dataset[DatasetSplit.TRAIN].features, dataset[DatasetSplit.TRAIN].target)
    validation_pool = lightgbm.Dataset(
        dataset[DatasetSplit.VALIDATION].features, dataset[DatasetSplit.VALIDATION].target, reference=train_pool
    )
    training_params = {
        'objective': 'binary',
        'metrics': ['binary_logloss', 'auc', 'binary_error'],
        'num_iterations': 1000,
        'seed': random_seed
    }
    with mlflow.start_run() as mlflow_run:
        model = lightgbm.train(training_params, train_pool, valid_sets=[validation_pool])
        test_metrics = _calculate_binary_classification_metrics(
            dataset[DatasetSplit.TEST].target,
            model.predict(dataset[DatasetSplit.TEST].features)  # type: ignore[arg-type]
        )
        mlflow.log_metrics(test_metrics)
        logger.info(f'Test metrics: {test_metrics}')
        return mlflow_run


@prefect.task
def register_model(
    mlflow_run: mlflow.entities.Run, model_name: str, model_alias: str
) -> mlflow.entities.model_registry.ModelVersion:
    model_uri = f'runs:/{mlflow_run.info.run_id}/model'
    model_version = mlflow.register_model(model_uri, model_name)
    mlflow_client = mlflow.MlflowClient()
    mlflow_client.set_registered_model_alias(model_name, model_alias, model_version.version)
    return model_version


@prefect.flow
def train_model(model_name: str, model_alias: str) -> None:
    set_up_logger(logger)

    dataset_dir = get_dataset_dir()
    dataset = load_dataset(dataset_dir)
    mlflow_run = train_model_inner(dataset)
    logger.info(f'run_id: {mlflow_run.info.run_id}')

    model_version = register_model(mlflow_run, model_name, model_alias)
    logger.info(
        f'model_version: name: {model_version.name} version: {model_version.version} aliases: {model_version.aliases}'
    )
