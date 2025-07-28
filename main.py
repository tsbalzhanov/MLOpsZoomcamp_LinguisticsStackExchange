import prefect

from data import download_and_unpack_archive, prepare_dataset
from train_model import train_model


def main() -> None:
    prefect.serve(
        download_and_unpack_archive.to_deployment('download-and-unpack_archive'),
        prepare_dataset.to_deployment('prepare-dataset'),
        train_model.to_deployment('train-model')
    )


if __name__ == '__main__':
    main()
