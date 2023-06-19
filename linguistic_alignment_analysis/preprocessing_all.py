import pandas as pd

from Helpers import read_csv, df_to_object, print_t, print_i
from Models.Discussion import Discussion
from Models.Post import Post
from datetime import datetime
import copy
import math
import re
from linguistic_alignment_analysis.compute_lexical_word_alignment import preprocess_message_lexical_word, \
    jaccard_overlap, adapted_LLA

# __jaccard_similarity_thread__ = 'AlignmentData/jaccard_similarity_thread.csv'
# __jaccard_similarity_linear__ = 'AlignmentData/jaccard_similarity_linear.csv'

__adapted_LLA_linear__ = 'AlignmentData/preprocessing_LLA_alignment_linear.csv'


def get_discusssion_posts(input_df):
    """
    Extracts the data from the dataframe and converts it into a dict of discussions with dicts of posts.

    :param input_df: dataframe with discussions
    :return: discussions as list of objects
    """

    print_t('getting data from dataframe and converting to discussions with posts')

    discussions = df_to_object(input_df)
    print_i('getting discussions completed')
    return discussions


def replace_urls(discussions):
    """
    Replaces URLs with [URL] tag
    :param discussions: discussions as list of objects
    :return: discussions as list of objects, with urls replaced in post messages
    """
    print_t('replacing URLs with [URL] tags')
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            message = re.sub('http[s]?://\S+', '[URL]', post.message)
            message = re.sub('www.\S+', '[URL]', message)
            post.update_message(message)
    print_i('replaced URLs with [URL] tags')
    return discussions


def get_discussion_threads(discussions):
    """
    Converts discussions into tree-structured threads
    :param discussions: discussions as list of objects
    :return: discussions with threads included in posts (based on parent posts)
    """
    print_t('getting threaded discussions')
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

    print_i('got threaded discussions')
    return discussions


def get_discussion_linear_threads(discussions):
    """
    Converts discussions into linear threads
    :param discussions: discussions as list of objects
    :return: discussions as list of objects with threads included in posts (based on time)
    """
    print_t('getting linear discussions')
    # posts are with post_id already ordered by date in discussions
    # create thread with all previous posts
    for i in discussions.keys():
        discussion = discussions[i]
        post_list = list(discussion.posts.keys())
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            history = post_list[:post_list.index(j)]
            post.set_thread(history)

    print_i('got linear discussions')
    return discussions


def remove_empty_discussions(discussions):
    """
    Removes discussions with emtpy messages
    :param discussions: discussions as list of objects
    :return: discussions as list of objects, with discussions with empty messages removed.
    """
    print_t('removing discussions with empty messages')

    empty_discussion_indices = []
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            text_ = post.message
            if not text_ or not isinstance(text_, str):
                # message has no text, remove discussion.
                empty_discussion_indices.append(discussion.discussion_id)

    print_i('removing ' + str(len(empty_discussion_indices)) + 'discussions...')
    stripped_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in empty_discussion_indices }

    print_i('removed empty discussions, ' + str(len(stripped_discussions.keys())) + ' discussions left')
    return stripped_discussions


def merge_consecutive_messages(discussions):
    """
    Merges consecutive messages
    :param discussions: discussions as list of objects
    :return: discussions as list of objects, with consecutive messages of a same author merged.
    """
    print_t('merging consecutive messages')
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

    print_i('merged consecutive messages')
    return discussions


def remove_0_overlap(discussions, avg_overlap_data_path):
    """
    Removes discussions where the average Jaccard similarity is zero (outliers)
    :param discussions: discussions as list of objects
    :param avg_overlap_data_path: path where to store the average overlap
    :return: discussions as list of objects with average of 0 overlap removed
    """

    # {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
    # """
    # Run jaccard preprocessing
    print_t('preprocessing messages')
    preprocessed_messages = {}
    # get all the preprocessed posts
    for i in discussions.keys():
        discussion = discussions[i]
        print_i('preprocessing for discussion' + str(discussion.discussion_id))
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_lexical_word(post.message)
            preprocessed_messages[str(i) + '-' + str(j)] = preprocessed

    print_i('preprocessed messages')

    print_t('computing Jaccard similarity for all messages and all of their parents')
    data = []
    for i in discussions.keys():
        print('computing overlap: ', i)
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index]
            for k in range(0, len(post.thread)):
                initial_post_id = post.thread[k]
                initial_preprocessed_index = str(discussion.discussion_id) + '-' + str(initial_post_id)
                initial_preprocessed = preprocessed_messages[initial_preprocessed_index]
                alignment = jaccard_overlap(initial_preprocessed, response_preprocessed)
                distance = len(post.thread) - k
                data.append([
                    discussion.discussion_id,
                    initial_post_id,
                    post.post_id,
                    distance,
                    alignment
                ])
    print_i('computed jaccard similarity')

    # Obtain jaccard similarity dataframe
    discussions_df = pd.DataFrame(data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'jaccard'])

    # Compute averages for each discussion
    print_t('computing averages')
    averages = []
    discussion_indices = discussions_df['discussion_id'].unique()
    for i in discussion_indices:
        print('averaging alignment', i)
        discussion_df = discussions_df.loc[discussions_df['discussion_id'] == i]
        discussion_alignment_avg = discussion_df['jaccard'].mean()
        averages.append([
            i,
            discussion_alignment_avg
        ])

    # Store average overlap
    average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_alignment'])
    average_df.to_csv(avg_overlap_data_path)
    print_i('Computed averages')

    # """
    # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

    # average_df = read_csv(avg_overlap_data_path)
    # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

    # Remove discussions with overlap of 0
    print_t('Removing 0 overlap discussions')
    discussion_0 = average_df[average_df['average_alignment'] == 0]
    discussion_ids_0 = discussion_0['discussion_id'].unique().tolist()
    print_i('Found ' + str(len(discussion_ids_0)) + 'discussions to remove')
    filtered_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in discussion_ids_0 }

    print_i('Removed 0 overlap discussions')
    return filtered_discussions


def remove_high_overlap(discussions, avg_overlap_data_path):
    """
    Removes discussions where the average adapted LLA is higher than 0.5 (outliers)
    :param discussions: discussions as list of objects
    :param avg_overlap_data_path: path from where to read the average overlap
    :return: discussions as list of objects with average higher than 0.2 overlap
    """
    _alignment_threshold_ = 0.5
    print_t('Removing high overlap')

    # Run adapted LLA preprocessing
    print_t('preprocessing messages')
    preprocessed_messages = {}
    # get all the preprocessed posts
    for i in discussions.keys():
        discussion = discussions[i]
        print_i('preprocessing for discussion' + str(discussion.discussion_id))
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_lexical_word(post.message)
            preprocessed_messages[str(i) + '-' + str(j)] = preprocessed

    print_i('preprocessed messages')

    print_t('computing adapted LLA for all messages and all of their parents')
    data = []
    for i in discussions.keys():
        print('computing overlap: ', i)
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index]
            for k in range(0, len(post.thread)):
                initial_post_id = post.thread[k]
                initial_preprocessed_index = str(discussion.discussion_id) + '-' + str(initial_post_id)
                initial_preprocessed = preprocessed_messages[initial_preprocessed_index]
                alignment = adapted_LLA(initial_preprocessed, response_preprocessed)
                distance = len(post.thread) - k
                data.append([
                    discussion.discussion_id,
                    initial_post_id,
                    post.post_id,
                    distance,
                    alignment
                ])
    print_i('computed adapted LLA')

    # Obtain adapted LLA dataframe
    discussions_df = pd.DataFrame(data,
                                  columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance',
                                           'adapted_LLA'])

    # Compute averages for each discussion
    print_t('computing averages')
    averages = []
    discussion_indices = discussions_df['discussion_id'].unique()
    for i in discussion_indices:
        print('averaging alignment', i)
        discussion_df = discussions_df.loc[discussions_df['discussion_id'] == i]
        discussion_alignment_avg = discussion_df['adapted_LLA'].mean()
        averages.append([
            i,
            discussion_alignment_avg
        ])

    # Store average overlap
    average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_alignment'])
    average_df.to_csv(avg_overlap_data_path)
    print_i('Computed averages')

    # If removing 0 overlap is run before, overlap was already computed. Therefore, loads data
    # average_df = read_csv(avg_overlap_data_path)

    discussion_high = average_df[average_df['average_alignment'] > 0.5]
    discussion_ids_high = discussion_high['discussion_id'].unique().tolist()
    print_i('Found ' + str(len(discussion_ids_high)) + 'discussions to remove')
    filtered_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in discussion_ids_high }

    print_i('Removed high overlap discussions')
    return filtered_discussions


def run_preprocessing(datapath):
    """
    Main function for running the preprocessing on a csv containing all discussions, posts, date, author, parent post etc.
    :param datapath: path where the csv data is read from
    :return: two lists of discussions as objects, for threads and for linear.
    """
    data = read_csv(datapath)
    # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    """
    unique_discussions = data['discussion_id'].unique().tolist()
    unique_discussions_shorter = unique_discussions[:50]
    data_shorter = data[data['discussion_id'].isin(unique_discussions_shorter)]
    discussion_posts = get_discusssion_posts(data_shorter)
    """
    # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    discussion_posts = get_discusssion_posts(data)
    # }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    removed_empty = remove_empty_discussions(discussion_posts)
    replaced_urls = replace_urls(removed_empty)

    # threads = get_discussion_threads(replaced_urls)
    # threads_consecutive_merged = merge_consecutive_messages(copy.deepcopy(replaced_urls))
    # threads_removed_0_overlap = remove_0_overlap(threads_consecutive_merged, __jaccard_similarity_thread__)
    # threads_removed_high_overlap = remove_high_overlap(threads_removed_0_overlap, __jaccard_similarity_thread__)

    linear = get_discussion_linear_threads(copy.deepcopy(replaced_urls))
    linear_consecutive_merged = merge_consecutive_messages(linear)
    # linear_removed_0_overlap = remove_0_overlap(linear_consecutive_merged, __jaccard_similarity_linear__)
    linear_removed_high_overlap = remove_high_overlap(linear_consecutive_merged, __adapted_LLA_linear__)

    return linear_removed_high_overlap
