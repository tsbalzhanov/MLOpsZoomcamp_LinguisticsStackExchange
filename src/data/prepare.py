import enum
import logging
import pathlib

import bs4
import pandas as pd
import prefect
import sklearn.model_selection as sk_model_selection  # type: ignore[import-untyped]

from common import (
    TARGET_COLUMN, DatasetSplit, StackExchangeDataset, StackExchangePool, prepare_output_dir, set_up_logger
)

from .data_common import get_raw_data_dir


logger = logging.getLogger(__name__)


class PostType(enum.IntEnum):
    POST = 1
    COMMENT = 2


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
    return 1 + sum(sym == '|' for sym in tags.strip('|'))


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


def _extract_target(posts_df: pd.DataFrame) -> StackExchangePool:
    return StackExchangePool(
        posts_df.drop(columns=[TARGET_COLUMN]).astype(pd.Float32Dtype()),
        posts_df[TARGET_COLUMN]
    )


@prefect.task
def prepare_dataset_inner(
    data_dir: pathlib.Path, validation_size: float, test_size: float, random_seed: int
) -> dict[DatasetSplit, StackExchangePool]:
    set_up_logger(logger)

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
    return {
        DatasetSplit.TRAIN: _extract_target(train_df),
        DatasetSplit.VALIDATION: _extract_target(validation_df),
        DatasetSplit.TEST: _extract_target(test_df)
    }


@prefect.task
def save_dataset(dataset: StackExchangeDataset, dataset_dir: pathlib.Path) -> None:
    for split, pool in dataset.items():
        pool.features.to_parquet(dataset_dir / f'{split}_features.parquet')
        pool.target.to_frame().to_parquet(dataset_dir / f'{split}_target.parquet')


@prefect.flow
def prepare_dataset(
    root_data_dir: pathlib.Path, random_seed: int, validation_size: float, test_size: float, overwrite: bool = False
) -> None:
    set_up_logger(logger)
    raw_data_dir = get_raw_data_dir(root_data_dir)
    assert raw_data_dir.exists()
    dataset_dir = root_data_dir / 'dataset'
    prepare_output_dir(dataset_dir, overwrite, logger)
    dataset = prepare_dataset_inner(raw_data_dir, validation_size, test_size, random_seed)
    save_dataset(dataset, dataset_dir)
