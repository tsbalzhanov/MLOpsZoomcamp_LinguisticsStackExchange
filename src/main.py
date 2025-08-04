import prefect

from data import download_and_unpack_archive, prepare_dataset
from model import prepare_mlserver_resources, train_model


def main() -> None:
    prefect.serve(
        download_and_unpack_archive.to_deployment('download-and-unpack-archive'),  # type: ignore[arg-type]
        prepare_dataset.to_deployment('prepare-dataset'),  # type: ignore[arg-type]
        train_model.to_deployment('train-model'),  # type: ignore[arg-type]
        prepare_mlserver_resources.to_deployment('prepare-mlserver-resources')  # type: ignore[arg-type]
    )


if __name__ == '__main__':
    main()
