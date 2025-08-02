import pathlib


def get_raw_data_dir(root_data_dir: pathlib.Path) -> pathlib.Path:
    return root_data_dir / 'raw_data'
