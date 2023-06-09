import pandas as pd
from datetime import datetime

from Models.Discussion import Discussion
from Models.Post import Post


def read_csv(path):
    """
    Reads discussions into pandas
    :param path: path of the discussions csv
    :return: discussions in dataframe
    """
    print_t('reading csv from: ' + str(path))
    df = pd.read_csv(path)
    print_i('reading csv completed')
    return df


def color_scheme():
    return ['#d74a94', '#6bb2a5', '#fdd516', '#77b75b', '#ff8800']


def print_t(message):
    print('[TASK]', message)


def print_i(message):
    print('[INFO]', message)


def df_to_object(df):
    """
    Converts discussion dataframe to list of objects
    :param df: discussion dataframe
    :return: list of discussion objects
    """
    discussions = {}
    # Divide into discussions & posts
    discussion_indices = df['discussion_id'].unique()
    for i in discussion_indices:
        discussion_df = df.loc[df['discussion_id'] == i]
        posts = {}
        for index, row in discussion_df.iterrows():
            date = datetime.strptime(str(row['creation_date']), "%Y-%m-%d %H:%M:%S")
            post = Post(row['discussion_id'], row['post_id'], row['text'], row['parent_post_id'], row['author_id'], date)
            posts[row['post_id']] = post
        discussion = Discussion(i, posts)
        discussions[i] = discussion

    return discussions



def object_to_df(objects):
    """
    Converts list of objects to discussion dataframe
    :param objects: list of discussion objects
    :return: discussion dataframe
    """