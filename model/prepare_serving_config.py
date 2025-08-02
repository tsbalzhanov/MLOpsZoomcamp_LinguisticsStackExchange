import json
import logging
import pathlib

import mlflow
import prefect


from common import set_up_logger


logger = logging.getLogger(__name__)


def get_mlserver_dir(root_data_dir: pathlib.Path) -> pathlib.Path:
    return root_data_dir / 'mlserver'


@prefect.flow
def prepare_serving_config(
    root_data_dir: pathlib.Path, mlflow_tracking_uri: str, model_name: str, model_alias: str
) -> None:
    set_up_logger(logger)
    mlserver_dir = get_mlserver_dir(root_data_dir)
    mlserver_dir.mkdir(exist_ok=True)

    mlflow_client = mlflow.MlflowClient(mlflow_tracking_uri)
    model_version = mlflow_client.get_model_version_by_alias(model_name, model_alias)
    logger.info(f'Model version: {model_version}')

    model_uri = mlflow_client.get_model_version_download_uri(model_name, model_version.version)
    logger.info(f'Model uri: {model_uri}')

    model_settings = {
        'name': model_name,
        'implementation': 'mlserver_lightgbm.LightGBMModel',
        'parameters': {
            'uri': f'{model_uri}/model.lgb',
            'version': model_version.version
        }
    }
    (mlserver_dir / 'model-settings.json').write_text(json.dumps(model_settings, ensure_ascii=False, indent=4))
