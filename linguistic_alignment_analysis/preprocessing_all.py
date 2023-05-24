import pandas as pd

from Helpers import read_csv
from Models.Discussion import Discussion
from Models.Post import Post
from datetime import datetime
import copy
import math
import re


__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_two_posts.csv'


"""
Extracts the data from the dataframe and converts it into a dict of discussions with dicts of posts.
"""
def get_discusssion_posts(input_df):
    print('[TASK] getting data from dataframe and converting to discussions with posts')

    discussions = {}
    # Divide into discussions & posts
    discussion_indices = input_df['discussion_id'].unique()
    # discussion_indices = [1, 2, 3] # For testing purposes
    for i in discussion_indices:
        discussion_df = input_df.loc[input_df['discussion_id'] == i]
        posts = {}
        for index, row in discussion_df.iterrows():
            date = datetime.strptime(str(row['creation_date']), "%Y-%m-%d %H:%M:%S")
            post = Post(row['discussion_id'], row['post_id'], row['text'], row['parent_post_id'], row['author_id'], date)
            posts[row['post_id']] = post
        discussion = Discussion(i, posts)
        discussions[i] = discussion
    print('[INFO] task completed')
    return discussions


"""
Replace URLs with [URL] tag
"""
def replace_urls(discussions):
    print('[TASK] replacing URLs with [URL] tags')
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            message = re.sub('http[s]?://\S+', '[URL]', post.message)
            message = re.sub('www.\S+', '[URL]', message)
            post.update_message(message)
    print('[INFO] task completed')
    return discussions


"""
Converts discussions into tree-structured threads
"""
def get_discussion_threads(discussions):
    print('[TASK] getting threaded discussions')
    # Make the threads
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            if not math.isnan(post.parent_id):
                parent_post = discussion.posts[post.parent_id]
                thread = parent_post.thread.copy()
                thread.append(parent_post.post_id)
                post.set_thread(thread)

    print('[INFO] task completed')
    return discussions

"""
Converts discussions into linear threads
"""
def get_discussion_linear_threads(discussions):
    print('[TASK] getting linear discussions')
    # posts are with post_id already ordered by date in discussions
    # create thread with all previous posts
    for i in discussions.keys():
        discussion = discussions[i]
        post_list = list(discussion.posts.keys())
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            history = post_list[:post_list.index(j)]
            post.set_thread(history)

    print('[INFO] task completed')
    return discussions


# """
# Removes empty messages & threads in which these posts are
# """
# def remove_empty(discussions):
#     print('[TASK] removing empty messages & threads')
#     counter = 0
#     for i in discussions.keys():
#         empty_messages = []
#         discussion = discussions[i]
#         for j in discussion.posts.keys():
#             post = discussion.posts[j]
#             text_ = post.message
#             if not text_ or not isinstance(text_, str):
#                 # This message has no text
#                 print(str(counter), 'posts found without text!', post.post_id, j)
#                 empty_messages.append(j)
#                 counter += 1
#
#         if len(empty_messages) > 0:
#             print('empty messages: ', empty_messages)
#             print(discussion.posts)
#             discussion.posts = [post for post in discussion.posts if set(empty_messages).isdisjoint(post.thread)]
#             print(discussion.posts)
#     return discussions


"""
Removes discussions with emtpy messages
"""
def remove_empty_discussions(discussions):
    print('[TASK] removing discussions with empty messages')

    empty_discussion_indices = []
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            text_ = post.message
            if not text_ or not isinstance(text_, str):
                # message has no text, remove discussion.
                empty_discussion_indices.append(discussion.discussion_id)

    print('[INFO] removing ', len(empty_discussion_indices), 'discussions...')
    stripped_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in empty_discussion_indices }

    print('[INFO] task completed, ', len(stripped_discussions.keys()), 'discussions left')
    return stripped_discussions


"""
Merges consecutive messages
"""
def merge_consecutive_messages(discussions):
    print('[TASK] merging consecutive messages')
    # Find message where previous message is of the same author
    for i in discussions.keys():
        discussion = discussions[i]
        to_remove_posts = []
        reversed_indices = reversed(list(discussion.posts.keys()))
        for j in reversed_indices:
            post = discussion.posts[j]
            if len(post.thread) > 0:
                previous_post_index = post.thread[-1]
                previous_post = discussion.posts[previous_post_index]
                if post.username == previous_post.username:
                    # Add message text to previous message
                    new_message = previous_post.message + post.message
                    previous_post.update_message(new_message)
                    # Keep track of this message id and the new (previous) id
                    to_remove_posts.append(j)

        # Replace all thread histories where that id occured with the previous message id to keep the threads intact
        for k in to_remove_posts:
            del discussion.posts[k]

        for j in discussion.posts.keys():
            post = discussion.posts[j]
            new_threads = [indx for indx in post.thread if indx not in to_remove_posts]
            post.set_thread(new_threads)

    print('[INFO] task completed')
    return discussions


"""
MAIN: run preprocessing
"""
def run_preprocessing():
    global __datapath__
    print(__datapath__)
    data = read_csv(__datapath__)
    discussion_posts = get_discusssion_posts(data)
    removed_empty = remove_empty_discussions(discussion_posts)
    replaced_urls = replace_urls(removed_empty)
    threads = get_discussion_threads(replaced_urls)
    linear = get_discussion_linear_threads(copy.deepcopy(replaced_urls))
    threads_consecutive_merged = merge_consecutive_messages(threads)
    linear_consecutive_merged = merge_consecutive_messages(linear)

    return threads_consecutive_merged, linear_consecutive_merged
