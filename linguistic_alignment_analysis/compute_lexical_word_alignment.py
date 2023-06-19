import copy
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

from Helpers import print_t, print_i, read_csv, store_data
import pickle


def get_wordnet_pos(treebank_tag):
    """
        Returns the wordnet version of the treebank POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None     # for easy if-statement


def preprocess_message_lexical_word(post):
    """
    Preprocesses text posts before applying LILLA, by tokenizing, lowercasing and removing separate punctuation tokens.
    :param post: message to preprocess
    :return: preprocessed message: list of lemmas
    """
    # Tokenize post
    tokens = word_tokenize(post)
    # Remove tokens that exist solely of punctuation and remove stopwords
    stop_words = set(stopwords.words('english'))
    post_punctuation_stopwords_removed = [token.lower() for token in tokens if token not in string.punctuation and token not in stop_words]
    # Apply POS
    tagged = nltk.pos_tag(post_punctuation_stopwords_removed)
    # Get lemmas
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            lemmas.append(lemmatizer.lemmatize(word))
        else:
            lemmas.append(lemmatizer.lemmatize(word, pos=wntag))

    return lemmas


def get_preprocessed_messages_for_lexical_word(discussions, path, author_data_path):
    """
    Preprocesses each message for this analysis
    :param discussions: discussions in list of objects
    :param path: path where to store dataframe pickle
    :param author_data_path: path where to store author data
    :return: dict with 'discussion.id-post.id' key and preprocessed message value
    """
    print_t('preprocessing messages')
    preprocessed_posts = {}
    # get all the preprocessed posts
    data = []
    for i in discussions.keys():
        print(f'Preprocessing for lexical word: {i} out of {len(discussions.keys())}')
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_lexical_word(post.message)
            preprocessed_posts[str(i) + '-' + str(j)] = preprocessed
            data.append([
                discussion.discussion_id,
                post.post_id,
                post.username,
                post.message,
                preprocessed
            ])

    print_i('task completed')

    print_t('storing preprocessing data')
    df = pd.DataFrame(data, columns=['discussion_id', 'post_id', 'author_id', 'text', 'preprocessed_text'])
    df.to_pickle(path)
    print_i('stored preprocessing data')

    # author_data = get_data_per_author(df)
    # print_t('Pickling data to ' + str(author_data_path))
    # store_file = open(author_data_path, 'ab')
    # pickle.dump(author_data, store_file)
    # store_file.close()
    # print_i('task completed')

    return preprocessed_posts


def jaccard_overlap(initial_message, response_message):
    """
    Alignment function: Jaccard index
    """
    cnt_initial = Counter()
    for word in initial_message:
        cnt_initial[word] += 1
    cnt_response = Counter()
    for word in response_message:
        cnt_response[word] += 1
    intersection = cnt_initial & cnt_response
    jaccard = intersection.total() / (len(initial_message) + len(response_message))
    return jaccard


def adapted_LLA(initial_message, response_message):
    """
    Alignment function: adapted LLA
    :param initial_message: list of lemmas of initial message
    :param response_message: list of lemmas of response message
    :return: adapted LLA
    """
    w_in_R_in_I = [lemma for lemma in response_message if lemma in initial_message]
    if len(w_in_R_in_I) != 0 and len(response_message) != 0:
        return len(w_in_R_in_I) / len(response_message)
    else:
        # print(initial_message)
        # print(response_message)
        return 0


def get_data_per_author(df):
    """
    Obtain all posts and words per authors
    """
    print_t('Getting data per author')
    unique_authors = df['author_id'].unique()
    author_general = {}
    for author in unique_authors:
        df_author = df[df['author_id'] == author]
        posts = []
        preprocessed = []
        for index, row in df_author.iterrows():
            posts.append(row['preprocessed_text'])
            preprocessed = preprocessed + row['preprocessed_text']
        author_general[author] = {
            'posts': posts,
            'words': preprocessed
        }
    print_i('Task completed')
    return author_general


def get_unique_words(discussions, preprocessed_messages):
    """
    Returns a set of all unique words in the corpora
    :param discussions: as list of objects
    :param preprocessed_messages: as object
    :return: pandas series with unique words
    """
    word_set = set()
    for i in discussions.keys():
        discussion = discussions[i]
        print('discussion: ', i)
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index]
            word_set.update(response_preprocessed)
    word_series = pd.Series(list(word_set))
    return word_series


def compute_lexical_word_alignment_SCP(discussions, preprocessed_messages, path, author_data_path, normed_data_path):
    """
    Computes the actual alignment for each message and each of it's parent messages
    :param discussions: list of discussion objects
    :param preprocessed_messages: object of preprocessed messages
    :param path: path where to store the alignment dataframe
    :param author_data_path: path where to read the author data
    :param normed_data_path: path where to store the intermediate counts
    :return: dataframe with alignment
    """
    print_t('computing lexical word alignment for all messages and all of their parents')

    print_t('Loading preprocessed data from pickle path' + str(author_data_path))
    store_file = open(author_data_path, 'rb')
    author_data = pickle.load(store_file)
    store_file.close()
    print_i('Task completed')

    unique_words = get_unique_words(discussions, preprocessed_messages)

    # proportion of words in R that are w
    # proportion of responses R by author i containing w
    # proportion of words by author i that are m

    print_t('initializing author lists and vars')
    # Amount of posts
    no_posts = len(preprocessed_messages.keys())
    post_series = pd.Series(preprocessed_messages.keys())

    # Amount of words
    no_words = len(unique_words)

    # Amount of authors
    no_authors = len(author_data.keys())
    author_series = pd.Series(author_data.keys())

    dtype = pd.SparseDtype('float', np.nan)

    a2 = pd.DataFrame(np.nan, post_series, unique_words)
    b1 = pd.DataFrame(0, unique_words, author_series)
    b2 = pd.DataFrame(0, unique_words, author_series)

    a2.astype(dtype)
    b1.astype(dtype)
    b2.astype(dtype)

    # init norm lists
    b1_posts = pd.Series(np.nan, author_series)
    b2_words = pd.Series(np.nan, author_series)
    for a_idx in author_data.keys():
        b1_posts[a_idx] = len(author_data[a_idx]['posts'])
        b2_words[a_idx] = len(author_data[a_idx]['words'])
    print_i('Author lists initialized')

    # Fill the counts
    print_t('Computing counts')
    for i in discussions.keys():
        discussion = discussions[i]
        print('discussion: ', i)
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            post_idx = str(discussion.discussion_id) + '-' + str(post.post_id)
            post_preprocessed = preprocessed_messages[post_idx]
            word_counter = Counter(post_preprocessed)
            len_post = len(post_preprocessed)
            for k in post_preprocessed:
                a2.at[post_idx, k] = word_counter[k] / len_post
                b1.at[k, post.username] += 1
                b2.at[k, post.username] += word_counter[k]
    print_i('Counts computed')
    # save the counts

    # compute the fractions
    b1 = b1 / b1_posts
    b2 = b2 / b2_words

    print_t('Storing normed counts')
    # Save the normed counts
    store_object = {
        'a2': a2,
        'b1': b1,
        'b2': b2
    }
    store_data(store_object, normed_data_path)
    print_i('Normed counts stored')

    # Init weight
    w1 = 0.5
    w2 = 0.5

    print_t('Computing alignment')
    alignment_data = []
    for i in discussions.keys():
        discussion = discussions[i]
        print_i(f'Discussion: {i} out of {len(discussions.keys())}')
        for j in discussion.posts.keys():
            post_response = discussion.posts[j]
            post_idx = f'{i}-{j}'
            post_response_preprocessed = preprocessed_messages[post_idx]
            for k in range(0, len(post_response.thread)):
                post_initial_id = post_response.thread[k]
                post_initial_preprocessed = preprocessed_messages[f'{i}-{post_initial_id}']

                alignments = []
                for w in post_response_preprocessed:
                    a1_waarde = int(w in post_initial_preprocessed)
                    a2_waarde = a2.at[post_idx, w]
                    b1_waarde = b1.at[w, post_response.username]
                    b2_waarde = b2.at[w, post_response.username]

                    scaled_SCP_w = ((a1_waarde * w1) + (a2_waarde * w2)) - ((b1_waarde * w1) + (b2_waarde * w2))
                    alignments.append(scaled_SCP_w)

                average_alignment = np.mean(alignments)
                distance = len(post_response.thread) - k
                alignment_data.append([
                    discussion.discussion_id,
                    post_initial_id,
                    post_response.post_id,
                    distance,
                    average_alignment
                ])

    # data = []
    # weight = np.array([0.5, 0.5])
    # for i in discussions.keys():
    #     discussion = discussions[i]
    #     print('discussion: ', i)
    #     for j in discussion.posts.keys():
    #         post = discussion.posts[j]
    #         response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
    #         response_preprocessed = preprocessed_messages[response_preprocessed_index]
    #
    #         word_response_counter = Counter(response_preprocessed)
    #         word_author_counter = Counter(author_data[post.username]['words'])
    #         response_length = len(response_preprocessed)
    #         no_words_by_author = len(author_data[post.username]['words'])
    #         no_posts_by_author = len(author_data[post.username]['posts'])
    #
    #         for k in range(0, len(post.thread)):
    #             initial_post_id = post.thread[k]
    #             initial_preprocessed_index = str(discussion.discussion_id) + '-' + str(initial_post_id)
    #             initial_preprocessed = preprocessed_messages[initial_preprocessed_index]
    #
    #             alignments = []
    #             for lemma in response_preprocessed:
    #                 if lemma in initial_preprocessed:
    #                     a_1 = 1
    #                 else:
    #                     a_1 = 0
    #
    #                 a_2 = word_response_counter[lemma] / response_length
    #
    #                 no_responses_with_word = len([post for post in author_data[post.username]['posts'] if lemma in post])
    #
    #                 b_1 = no_responses_with_word / no_posts_by_author
    #
    #                 b_2 = word_author_counter[lemma] / no_words_by_author
    #
    #                 a = np.array([a_1, a_2])
    #                 b = np.array([b_1, b_2])
    #
    #                 scaled_SCP_w = np.dot(a, weight) - np.dot(b, weight)
    #                 alignments.append(scaled_SCP_w)
    #             average_alignment = np.mean(alignments)
    #             distance = len(post.thread) - k
    #             data.append([
    #                 discussion.discussion_id,
    #                 initial_post_id,
    #                 post.post_id,
    #                 distance,
    #                 average_alignment
    #             ])
    print_i('computed alignment')

    print_t('storing alignment data')
    df = pd.DataFrame(alignment_data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'lexical_word_alignment'])
    df.to_csv(path)
    print_i('task completed')

    return df


def compute_lexical_word_alignment(discussions, preprocessed_messages, path):
    """
    Computes adapted LILLA alignment between messages in discussions
    :param discussions:
    :param preprocessed_messages:
    :param path:
    :return: df with alignment
    """
    print_t('computing lexical word alignment for all messages and all of their parents')
    data = []
    for i in discussions.keys():
        print('computing alignment', i)
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
    print('[INFO] task completed')

    print('[TASK] storing alignment data')
    df = pd.DataFrame(data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'lexical_word_alignment'])
    df.to_csv(path)
    print('[INFO] task completed')

    return df


def get_histograms_lexical_word_alignment_per_5(df):
    """
    Generates a histogram for all discussions individually, turns per alignment bin
    One histogram per 5 discussions
    """
    discussion_ids = list(df['discussion_id'].unique())

    while len(discussion_ids) > 0:
        if len(discussion_ids) > 5:
            plot_disc_ids = discussion_ids[:5]
        else:
            plot_disc_ids = discussion_ids

        first_index = plot_disc_ids[0]
        last_index = plot_disc_ids[-1]

        for d_idx in plot_disc_ids:
            discussion_df = df[df['discussion_id'] == d_idx]
            alignment = discussion_df['lexical_word_alignment'] * 2
            pyplot.hist(alignment, bins=np.arange(0, 1, 0.05), alpha=0.3, label=str(d_idx))
        pyplot.title('Lexical word alignment for discussions: ' + str(first_index) + ' - ' + str(last_index))
        pyplot.legend(loc='upper right')
        pyplot.xlim([0, 1])
        pyplot.ylim([0, 70000])
        pyplot.xlabel('Alignment as Jaccard overlap')
        pyplot.ylabel('# of posts')
        pyplot.savefig('./Results/Lexical_word_alignment/histo-' + str(first_index) + '-' + str(last_index))
        pyplot.show()

        discussion_ids = discussion_ids[5:]
    # pyplot.show()



def get_overall_histogram_lexical_word_alignment(df):
    """
    Generates one histogram for all discussions individually, turns per alignment bin
    """
    discussion_ids = list(df['discussion_id'].unique())

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 300000) #300000 for linear
    # ax.set_yscale('log')

    ax.set_title('Lexical word alignment for all discussions')
    ax.set_xlabel('Alignment as Jaccard overlap')
    ax.set_ylabel('# of posts')

    # alignment is in range [0, 0.5], normalize to [0, 1]
    # df['lexical_word_alignment'] = (df['lexical_word_alignment'] + 1) / 2

    for d_idx in discussion_ids:
        discussion_df = df[df['discussion_id'] == d_idx]
        alignment = discussion_df['lexical_word_alignment']
        ax.hist(alignment, bins=np.arange(-1, 1, 0.025), alpha=0.1, label=str(d_idx)) #color='#d74a94'  histtype='step'
        print(d_idx)
    # sns.displot(df, x='lexical_word_alignment', hue='discussion_id', kde=True)

    fig.savefig('./Results/Lexical_word_alignment/all_histo_thread_stacked')
    fig.show()



def get_overall_histogram_lexical_word_alignment_stacked(df, path):
    """
    Generates one stacked histogram for all discussions combined, turns per alignment bin
    """
    # """
    print('mean: \t', df['lexical_word_alignment'].mean())
    print('median: \t', df['lexical_word_alignment'].median())
    print('min: \t', df['lexical_word_alignment'].min())
    print('max: \t', df['lexical_word_alignment'].max())
    print('percentiles: \t', df['lexical_word_alignment'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995, 1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Average overlap between posts')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(-1, 1)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(-1, 1)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(-1, -0.75)
    # ax3.set_ylim(0, 500)
    ax3.set_xlabel('Scaled SCP alignment')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(df['lexical_word_alignment'], bins=np.arange(-1, 1, 0.025),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(df['lexical_word_alignment'], bins=np.arange(-1, 1, 0.025),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(df['lexical_word_alignment'], bins=np.arange(-1, -0.75, 0.005),
             color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(path)
    fig.show()

    discussion_0 = df.loc[df['lexical_word_alignment'] == 0]
    discussion_ids_0 = discussion_0['discussion_id'].unique().tolist()
    print('Found 0: ', discussion_ids_0)

    discussions_low = df.loc[(df['lexical_word_alignment'] >= -1) & (df['lexical_word_alignment'] < 0.75)]
    discussion_ids_low = discussions_low['discussion_id'].unique().tolist()
    print('found low: ', discussion_ids_low)

    # """
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 30000) #2500000 for linear
    # ax.set_yscale('log')

    ax.set_title('Lexical word alignment for all discussions')
    ax.set_xlabel('Alignment from Scaled SCP')
    ax.set_ylabel('# of posts')

    # alignment is in range [-1, 1], normalize to [0, 1]
    # df['lexical_word_alignment'] = (df['lexical_word_alignment'] + 1) / 2

    alignment = df['lexical_word_alignment']
    ax.hist(alignment, bins=np.arange(-1, 1, 0.025), alpha=0.8, color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(path)
    fig.show()
    """

def get_average(discussions_df):
    """
    Computes the average overlap for each discussion
    :param discussions_df:
    :param path:
    :return: df with average overlap
    """

    averages = []
    discussion_indices = discussions_df['discussion_id'].unique()
    for i in discussion_indices:
        print('averaging alignment', i)
        discussion_df = discussions_df.loc[discussions_df['discussion_id'] == i]
        discussion_alignment_avg = discussion_df['lexical_word_alignment'].mean()
        averages.append([
            i,
            discussion_alignment_avg
        ])

    average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_alignment'])
    return average_df


def get_overall_alignment_stats_all_previous(align_df, storage_path):
    """
    Generates one histogram for the average alignment in all discussions
    :param align_df: dataframe containing alignment data
    :param storage_path: path where to store histogram
    """
    average_df = get_average(align_df)

    print('mean: \t', average_df['average_alignment'].mean())
    print('percentiles: \t', average_df['average_alignment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, 1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Average alignment in discussions, considering all previous messages')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(0, 1)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(0, 0.02)
    # ax3.set_ylim(0, 500)
    ax3.set_xlabel('Adapted LILLA alignment')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(average_df['average_alignment'], bins=np.arange(0, 0.03, 0.001),
             color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(storage_path)
    fig.show()


def get_overall_alignment_stats_consecutive(align_df, storage_path):
    """
    Generates one histogram for average alignment within consecutive messages in discussions
    :param align_df: dataframe containing alignment data
    :param storage_path: path where to store histogram
    """
    consecutive_df = align_df.loc[align_df['distance'] == 1]
    average_df = get_average(consecutive_df)

    print('mean: \t', average_df['average_alignment'].mean())
    print('percentiles: \t', average_df['average_alignment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, 1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Average alignment in discussions, considering only consecutive messages')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(0, 1)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(0, 0.02)
    # ax3.set_ylim(0, 500)
    ax3.set_xlabel('Adapted LILLA alignment')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(average_df['average_alignment'], bins=np.arange(0, 0.03, 0.001),
             color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(storage_path)
    fig.show()


def get_overall_alignment_stats_initial(align_df, storage_path):
    """
    Generates one histogram for average alignment with messages and initial message in discussions
    :param align_df: dataframe containing alignment data
    :param storage_path: path where to store histogram
    """
    consecutive_df = align_df.loc[align_df['initial_message_id'] == 1]
    average_df = get_average(consecutive_df)

    print('mean: \t', average_df['average_alignment'].mean())
    print('percentiles: \t', average_df['average_alignment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, 1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Average alignment in discussions, considering pairs with only the first message')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(0, 1)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(0, 0.02)
    # ax3.set_ylim(0, 500)
    ax3.set_xlabel('Adapted LILLA alignment')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(average_df['average_alignment'], bins=np.arange(0, 0.03, 0.001),
             color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(storage_path)
    fig.show()



# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i'], ['e', 'b', 'c', 'd', 'j', 'g', 'h', 'i'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], ['e', 'f', 'a'])

# print(jaccard_overlap(['a', 'b', 'c', 'd'], ['x', 'y', 'z', 'a']))
# print(jaccard_overlap(['a', 'b', 'c', 'd'], ['a', 'q', 'r', 's', 'x', 'y', 'a', 'z']))

# print(adapted_LLA(['a', 'b', 'c', 'd'], ['e', 'f', 'a']))
# print(adapted_LLA(['a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd']))
# print(adapted_LLA(['a', 'b', 'c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], ['e', 'f', 'a']))
