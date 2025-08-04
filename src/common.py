import dataclasses
import enum
import logging
import os
import pathlib
import shutil
import sys

import pandas as pd


TARGET_COLUMN = 'has_accepted_answer'


def set_up_logger(logger: logging.Logger) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def get_root_data_dir() -> pathlib.Path:
    data_dir = pathlib.Path(os.environ['DATA_DIR'])
    return create_dir(data_dir)


def get_dataset_dir() -> pathlib.Path:
    return create_dir(get_root_data_dir() / 'dataset_dir')


def create_dir(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        dir_permissions = 0o700
        path.mkdir(dir_permissions, parents=True)
    elif not path.is_dir():
        raise ValueError(f'Path `{path}` is not a directory')
    return path


def get_random_seed() -> int:
    return int(os.environ['RANDOM_SEED'])


def prepare_output_dir(output_dir: pathlib.Path, overwrite: bool, logger: logging.Logger) -> None:
    DIR_PERMISSIONS = 0o700
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            logger.warning(f'Output path `{output_dir}` already exists')
            return
    output_dir.mkdir(DIR_PERMISSIONS)


@dataclasses.dataclass
class StackExchangePool:
    features: pd.DataFrame
    target: pd.Series


class DatasetSplit(enum.StrEnum):
    TRAIN = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()


StackExchangeDataset = dict[DatasetSplit, StackExchangePool]
