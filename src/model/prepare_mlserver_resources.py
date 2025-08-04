import json
import logging
import os
import pathlib
import shutil

import mlflow
import prefect

from common import set_up_logger


logger = logging.getLogger(__name__)


def get_mlserver_dir() -> pathlib.Path:
    return pathlib.Path(os.environ['MLSERVER_DATA_DIR'])


@prefect.flow
def prepare_mlserver_resources(model_name: str, model_alias: str) -> None:
    set_up_logger(logger)
    mlserver_dir = get_mlserver_dir()
    # cleanup mlserver_dir dir
    for item in mlserver_dir.iterdir():
        shutil.rmtree(item) if item.is_dir() else item.unlink()

    mlflow_client = mlflow.MlflowClient()
    model_version = mlflow_client.get_model_version_by_alias(model_name, model_alias)
    logger.info(f'Model version: {model_version}')

    model_uri = mlflow_client.get_model_version_download_uri(model_name, model_version.version)
    logger.info(f'Model uri: {model_uri}')

    mlflow.artifacts.download_artifacts(model_uri, dst_path=str(mlserver_dir))
    (mlserver_dir / 'artifacts').rename(mlserver_dir / model_name)

    model_settings = {
        'name': model_name,
        'implementation': 'mlserver_lightgbm.LightGBMModel',
        'parameters': {
            'uri': f'{model_name}/model.lgb',
            'version': model_version.version
        }
    }
    (mlserver_dir / 'model-settings.json').write_text(json.dumps(model_settings, ensure_ascii=False, indent=4))
