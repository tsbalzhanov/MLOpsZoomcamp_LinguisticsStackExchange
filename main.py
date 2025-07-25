import dataclasses
import enum
import logging
import pathlib
import sys
import tempfile

import bs4
import click
import lightgbm
import mlflow
import mlflow.entities
import mlflow.lightgbm
import numpy as np
import pandas as pd
import pyunpack
import requests
import sklearn.metrics as sk_metrics  # type: ignore[import-untyped]
import sklearn.model_selection as sk_model_selection  # type: ignore[import-untyped]
import tqdm


logger = logging.Logger(__name__)


def set_up_logger() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class PostType(enum.IntEnum):
    POST = 1
    COMMENT = 2


@dataclasses.dataclass
class StackExchangeDataset:
    features: pd.DataFrame
    target: pd.Series


def get_download_url(community: str) -> str:
    return f'https://archive.org/download/stackexchange/{community}.stackexchange.com.7z'


def get_model_name(community: str) -> str:
    return f'stack_exchange_{community}_post_acceptance_classifier'


def download_and_unpack_archive(url: str, output_dir_path: pathlib.Path | str) -> None:
    output_dir_path = pathlib.Path(output_dir_path)
    output_dir_path.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = pathlib.Path(temp_dir) / 'file'
        with temp_file_path.open('wb') as temp_file:
            logger.info(f'Downloading {url}')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            response_size = int(response.headers.get('content-length', 0))
            with tqdm.tqdm(total=response_size, unit='iB', unit_scale=True, unit_divisor=2**10) as bar:
                for data in response.iter_content(chunk_size=512 * 2**10):
                    temp_file.write(data)
                    bar.update(len(data))
        pyunpack.Archive(str(temp_file_path)).extractall(str(output_dir_path))


def _str_series_has_value(series: pd.Series) -> pd.Series:
    return ~series.isna() & (series.str.len() > 0)


def _get_html_text_length(html_markup: str | pd.api.typing.NAType) -> int:
    if pd.isna(html_markup):
        return 0
    bs = bs4.BeautifulSoup(html_markup, features='html.parser')
    return len(bs.get_text())


def _get_num_tags(tags: str | pd.api.typing.NAType) -> int:
    if pd.isna(tags) or len(tags) == 0:
        return 0
    return 1 + sum(map(lambda sym: sym == '|', tags.strip('|')))


def _load_and_parse_dataframe(
    file_path: pathlib.Path, dtypes: dict, *, date_fields: list[str] | None = None
) -> pd.DataFrame:
    df = pd.read_xml(file_path, dtype=dtypes)
    for date_field in date_fields or []:
        df[date_field] = pd.to_datetime(df[date_field], errors='coerce', format='ISO8601')
    return df


def _load_users_df(data_dir: pathlib.Path) -> pd.DataFrame:
    return _load_and_parse_dataframe(
        data_dir / 'Users.xml',
        {
            'DisplayName': pd.StringDtype(),
            'WebsiteUrl': pd.StringDtype(),
            'Location': pd.StringDtype(),
            'AboutMe': pd.StringDtype(),
            'AccountId': 'Int64'
        },
        date_fields=['CreationDate', 'LastAccessDate']
    ).rename(
        columns={
            'Id': 'user_id', 'Reputation': 'user_reputation', 'WebsiteUrl': 'website_url', 'AboutMe': 'about_me'
        }
    )[['user_id', 'user_reputation', 'website_url', 'about_me']]


def _prepare_users_df(users_df: pd.DataFrame) -> pd.DataFrame:
    users_df = users_df[users_df['user_id'] != -1].copy()
    users_df['has_website_url'] = _str_series_has_value(users_df['website_url'])
    users_df['has_description'] = _str_series_has_value(users_df['about_me'])
    return users_df.drop(columns=['website_url', 'about_me'])


def _load_all_posts_df(data_dir: pathlib.Path) -> pd.DataFrame:
    return _load_and_parse_dataframe(
        data_dir / 'Posts.xml', {
            'Body': pd.StringDtype(),
            'Title': pd.StringDtype(),
            'Tags': pd.StringDtype(),
            'ContentLicense': pd.StringDtype(),
            'OwnerUserId': 'Int64',
            'OwnerDisplayName': pd.StringDtype(),
            'ParentId': 'Int64',
            'ViewCount': 'Int64',
            'AnswerCount': 'Int64',
            'AcceptedAnswerId': 'Int64',
            'LastEditorUserId': 'Int64',
            'FavoriteCount': 'Int64'
        },
        date_fields=['CreationDate', 'LastActivityDate', 'LastEditDate', 'CommunityOwnedDate', 'ClosedDate']
    ).rename(
        columns={
            'Id': 'id', 'PostTypeId': 'post_type_id',
            'AcceptedAnswerId': 'accepted_answer_id', 'OwnerUserId': 'user_id', 'ParentId': 'parent_id',
            'Body': 'body', 'Title': 'title', 'Tags': 'tags', 'Score': 'score', 'FavoriteCount': 'favourite_count',
            'ViewCount': 'view_count', 'AnswerCount': 'answer_count', 'CommentCount': 'comment_count'
        }
    )[[
        'id', 'post_type_id', 'accepted_answer_id', 'user_id', 'parent_id', 'body', 'title', 'tags',
        'score', 'favourite_count', 'view_count', 'answer_count', 'comment_count'
    ]]


def _prepare_comments_df(all_posts_df: pd.DataFrame) -> pd.DataFrame:
    comments_df = all_posts_df[
        (all_posts_df['post_type_id'] == PostType.COMMENT)
    ][['parent_id', 'score', 'body', 'comment_count']].copy()
    comments_df['body_length'] = comments_df['body'].apply(_get_html_text_length)
    return comments_df.drop(columns=['body'])


def _prepare_posts_df(all_posts_df: pd.DataFrame) -> pd.DataFrame:
    posts_df = all_posts_df[
        (all_posts_df['post_type_id'] == PostType.POST) & (all_posts_df['user_id'] != -1)
    ].copy()
    posts_df['title_length'] = posts_df['title'].str.len()
    posts_df['body_length'] = posts_df['body'].apply(_get_html_text_length)
    posts_df['num_tags'] = posts_df['tags'].apply(_get_num_tags)
    posts_df['has_accepted_answer'] = posts_df['accepted_answer_id'].isna().astype(int)
    posts_df['favourite_count'] = posts_df['favourite_count'].fillna(0)
    return posts_df.drop(columns=['accepted_answer_id', 'title', 'body', 'tags', 'parent_id'])


def _extract_target(posts_df: pd.DataFrame) -> StackExchangeDataset:
    TARGET_COLUMN = 'has_accepted_answer'
    return StackExchangeDataset(
        posts_df.drop(columns=[TARGET_COLUMN]).astype(pd.Float32Dtype()),
        posts_df[TARGET_COLUMN]
    )


def prepare_dataframe(
    data_dir: pathlib.Path, validation_size: float, test_size: float, random_seed: int
) -> tuple[StackExchangeDataset, StackExchangeDataset, StackExchangeDataset]:
    logger.info('Preparing data for training')
    users_df = _prepare_users_df(_load_users_df(data_dir))
    all_posts_df = _load_all_posts_df(data_dir)
    posts_df = _prepare_posts_df(all_posts_df)
    comments_df = _prepare_comments_df(all_posts_df)

    posts_df = posts_df.merge(
        users_df.rename(columns={'has_website_url': 'user_has_website_url', 'has_description': 'user_has_description'}),
        how='left', on='user_id'
    ).drop(columns=['user_id'])
    posts_df = posts_df.merge(
        comments_df.rename(
            columns={
                'score': 'comment_score', 'body_length': 'comment_body_length', 'comment_count': 'comment_comment_count'
                }
            ),
        how='left', left_on='id', right_on='parent_id'
    ).drop(columns=['id', 'parent_id', 'post_type_id'])

    train_with_validation_df, test_df = sk_model_selection.train_test_split(
        posts_df, test_size=test_size, random_state=random_seed
    )
    train_df, validation_df = sk_model_selection.train_test_split(
        train_with_validation_df, test_size=validation_size / (1 - test_size), random_state=random_seed
    )
    return _extract_target(train_df), _extract_target(validation_df), _extract_target(test_df)


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


def train_model(
    train: StackExchangeDataset, validation: StackExchangeDataset, test: StackExchangeDataset, random_seed: int
) -> mlflow.entities.Run:
    logger.info('Training LightGBM model')
    mlflow.lightgbm.autolog()
    train_pool = lightgbm.Dataset(train.features, train.target)
    validation_pool = lightgbm.Dataset(validation.features, validation.target, reference=train_pool)
    training_params = {
        'objective': 'binary',
        'metrics': ['binary_logloss', 'auc', 'binary_error'],
        'num_iterations': 1000,
        'seed': random_seed
    }
    with mlflow.start_run() as mlflow_run:
        model = lightgbm.train(training_params, train_pool, valid_sets=[validation_pool])
        test_metrics = _calculate_binary_classification_metrics(test.target, model.predict(test.features))  # type: ignore[arg-type]
        mlflow.log_metrics(test_metrics)
        logger.info(f'Test metrics: {test_metrics}')
        return mlflow_run


@click.command()
@click.option(
    '--community', required=True, default='linguistics', help='Name of Stack Exchange community'
)
@click.option(
    '--root-data-dir',required=True,
    type=click.Path(
        file_okay=False, dir_okay=True, readable=True, writable=True, executable=True, path_type=pathlib.Path
    ),
    default=pathlib.Path('data'), help='Location of directory with data'
)
@click.option(
    '--random-seed', required=True, type=int, default=63, help='Seed for RNGs'
)
@click.option(
    '--validation-size', required=True, type=float, default=0.1, help='Validation dataset size'
)
@click.option(
    '--test-size', required=True, type=float, default=0.3, help='Validation dataset size'
)
@click.option(
    '--mlflow-tracking-uri', required=True, default='http://127.0.0.1:8900', help='MLFlow tracking URI'
)
def main(
    community: str, root_data_dir: pathlib.Path, random_seed: int, validation_size: float, test_size: float,
    mlflow_tracking_uri: str
) -> None:
    set_up_logger()
    DIR_PERMISSIONS = 0o700
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    root_data_dir.mkdir(DIR_PERMISSIONS, parents=True, exist_ok=True)
    raw_data_dir = root_data_dir / 'raw_data'
    raw_data_dir.mkdir(DIR_PERMISSIONS, exist_ok=True)
    download_and_unpack_archive(get_download_url(community), raw_data_dir)
    train, validation, test = prepare_dataframe(raw_data_dir, validation_size, test_size, random_seed)
    mlflow_run = train_model(train, validation, test, random_seed)
    logger.info(f'run_id: {mlflow_run.info.run_id}')


if __name__ == '__main__':
    main()
