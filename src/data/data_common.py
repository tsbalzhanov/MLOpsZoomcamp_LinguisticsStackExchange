import pathlib

from common import create_dir, get_root_data_dir


def get_raw_data_dir() -> pathlib.Path:
    return create_dir(get_root_data_dir() / 'raw_data')
