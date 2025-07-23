import enum
import pathlib

import bs4
import pandas as pd


class PostType(enum.IntEnum):
    POST = 1
    COMMENT = 2


def load_and_parse_dataframe(file_path: pathlib.Path | str, dtypes: dict, date_fields: list[str]) -> pd.DataFrame:
    df = pd.read_xml(file_path, dtype=dtypes)
    for date_field in date_fields:
        df[date_field] = pd.to_datetime(df[date_field], errors='coerce', format='ISO8601')
    return df


def str_series_has_value(series: pd.Series) -> pd.Series:
    return ~series.isna() & (series.str.len() > 0)


def load_users_df(data_dir: pathlib.Path) -> pd.DataFrame:
    return load_and_parse_dataframe(
        data_dir / 'Users.xml',
        {
            'DisplayName': pd.StringDtype(),
            'WebsiteUrl': pd.StringDtype(),
            'Location': pd.StringDtype(),
            'AboutMe': pd.StringDtype(),
            'AccountId': 'Int64'
        },
        ['CreationDate', 'LastAccessDate']
    ).rename(
        columns={
            'Id': 'user_id', 'Reputation': 'user_reputation', 'WebsiteUrl': 'website_url', 'AboutMe': 'about_me'
        }
    )[['user_id', 'user_reputation', 'website_url', 'about_me']]


def prepare_users_df(users_df: pd.DataFrame) -> pd.DataFrame:
    users_df = users_df[users_df['user_id'] != -1].copy()
    users_df['has_website_url'] = str_series_has_value(users_df['website_url'])
    users_df['has_description'] = str_series_has_value(users_df['about_me'])
    return users_df.drop(columns=['website_url', 'about_me'])


def load_all_posts_df(data_dir: pathlib.Path):
    return load_and_parse_dataframe(
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
        ['CreationDate', 'LastActivityDate', 'LastEditDate', 'CommunityOwnedDate', 'ClosedDate']
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


def get_html_text_length(html_markup: str | pd.api.typing.NAType) -> int:
    if pd.isna(html_markup):
        return 0
    bs = bs4.BeautifulSoup(html_markup, features='html.parser')
    return len(bs.get_text())


def get_num_tags(tags: str | pd.api.typing.NAType) -> int:
    if pd.isna(tags) or len(tags) == 0:
        return 0
    return 1 + sum(map(lambda sym: sym == '|', tags.strip('|')))


def prepare_comments_df(all_posts_df: pd.DataFrame) -> pd.DataFrame:
    comments_df = all_posts_df[
        (all_posts_df['post_type_id'] == PostType.COMMENT)
    ][['parent_id', 'score', 'body', 'comment_count']].copy()
    comments_df['body_length'] = comments_df['body'].apply(get_html_text_length)
    return comments_df.drop(columns=['body'])


def prepare_posts_df(all_posts_df: pd.DataFrame) -> pd.DataFrame:
    posts_df = all_posts_df[
        (all_posts_df['post_type_id'] == PostType.POST) & (all_posts_df['user_id'] != -1)
    ].copy()
    posts_df['title_length'] = posts_df['title'].str.len()
    posts_df['body_length'] = posts_df['body'].apply(get_html_text_length)
    posts_df['num_tags'] = posts_df['tags'].apply(get_num_tags)
    posts_df['has_accepted_answer'] = posts_df['accepted_answer_id'].isna().astype(int)
    posts_df['favourite_count'] = posts_df['favourite_count'].fillna(0)
    return posts_df.drop(columns=['accepted_answer_id', 'title', 'body', 'tags', 'parent_id'])


def main() -> None:
    data_dir = pathlib.Path('data')

    users_df = prepare_users_df(load_users_df(data_dir))
    all_posts_df = load_all_posts_df(data_dir)
    posts_df = prepare_posts_df(all_posts_df)
    comments_df = prepare_comments_df(all_posts_df)

    posts_df = posts_df.merge(
        users_df.rename(columns={'has_website_url': 'user_has_website_url', 'has_description': 'user_has_description'}),
        how='left', on='user_id'
    ).drop(columns=['user_id'])
    posts_df.head()
    posts_df = posts_df.merge(
        comments_df.rename(columns={'score': 'comment_score', 'body_length': 'comment_body_length', 'comment_count': 'comment_comment_count'}),
        how='left', left_on='id', right_on='parent_id'
    ).drop(columns=['parent_id'])
    print(posts_df.head())


if __name__ == '__main__':
    main()
