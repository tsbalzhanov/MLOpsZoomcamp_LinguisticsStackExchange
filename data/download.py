import logging
import pathlib
import tempfile

import prefect
import pyunpack
import requests
import tqdm

from common import prepare_output_dir, set_up_logger


logger = logging.getLogger(__name__)


def _get_download_url(community: str) -> str:
    return f'https://archive.org/download/stackexchange/{community}.stackexchange.com.7z'


@prefect.task
def download_and_unpack_archive_inner(url: str, output_dir_path: pathlib.Path | str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = pathlib.Path(temp_dir) / 'file'
        with temp_file_path.open('wb') as temp_file:
            logger.info(f'Downloading {url}')
            response = requests.get(url, stream=True, timeout=10.)
            response.raise_for_status()
            response_size = int(response.headers.get('content-length', 0))
            with tqdm.tqdm(total=response_size, unit='iB', unit_scale=True, unit_divisor=2**10) as bar:
                for data in response.iter_content(chunk_size=512 * 2**10):
                    temp_file.write(data)
                    bar.update(len(data))
        pyunpack.Archive(str(temp_file_path)).extractall(str(output_dir_path))


@prefect.flow
def download_and_unpack_archive(community: str, root_data_dir: pathlib.Path, overwrite: bool = False) -> None:
    set_up_logger(logger)
    dir_permissions = 0o700
    root_data_dir.mkdir(dir_permissions, parents=True, exist_ok=True)
    raw_data_dir = root_data_dir / 'raw_data'
    prepare_output_dir(raw_data_dir, overwrite, logger)
    download_and_unpack_archive_inner(_get_download_url(community), raw_data_dir)
