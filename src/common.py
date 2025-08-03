import dataclasses
import enum
import logging
import pathlib
import shutil
import sys

import pandas as pd


TARGET_COLUMN = 'has_accepted_answer'


def set_up_logger(logger: logging.Logger) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


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
