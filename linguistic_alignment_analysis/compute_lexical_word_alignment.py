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

from Helpers import print_t, print_i, read_csv
import pickle


"""
    Returns the wordnet version of the treebank POS tag
"""
def get_wordnet_pos(treebank_tag):
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


"""
    Preprocesses text posts before applying LILLA, by tokenizing, lowercasing and removing separate punctuation tokens.
    Params:
    - message (string)
    Returns:
    - preprocessed message (string[])
"""
def preprocess_message_lexical_word(post):
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

    print_t('storing alignment data')
    df = pd.DataFrame(data, columns=['discussion_id', 'post_id', 'author_id', 'text', 'preprocessed_text'])
    df.to_pickle(path)
    print_i('task completed')

    author_data = get_data_per_author(df)
    print_t('Pickling data to ' + str(author_data_path))
    store_file = open(author_data_path, 'ab')
    pickle.dump(author_data, store_file)
    store_file.close()
    print_i('task completed')

    return preprocessed_posts


"""
Alignment function: Jaccard index
"""
def jaccard_overlap(initial_message, response_message):
    cnt_initial = Counter()
    for word in initial_message:
        cnt_initial[word] += 1
    cnt_response = Counter()
    for word in response_message:
        cnt_response[word] += 1
    intersection = cnt_initial & cnt_response
    jaccard = intersection.total() / (len(initial_message) + len(response_message))
    return jaccard



"""
Obtain posts per authors
"""
def get_data_per_author(df):
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



"""
Computes the actual alignment for each message and each of it's parent messages
"""
def compute_lexical_word_alignment(discussions, preprocessed_messages, path, author_data_path):
    print_t('computing lexical word alignment for all messages and all of their parents')

    print_t('Loading preprocessed data from pickle path' + str(author_data_path))
    store_file = open(author_data_path, 'rb')
    author_data = pickle.load(store_file)
    store_file.close()
    print_i('Task completed')

    # proportion of words in R that are w
    # proportion of responses R by author i containing w
    # proportion of words by author i that are m

    # Amount of posts
    no_posts = len(preprocessed_messages.values())


    # Amount of words




    data = []
    weight = np.array([0.5, 0.5])
    for i in discussions.keys():
        discussion = discussions[i]
        print('discussion: ', i)
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index]

            word_response_counter = Counter(response_preprocessed)
            word_author_counter = Counter(author_data[post.username]['words'])
            response_length = len(response_preprocessed)
            no_words_by_author = len(author_data[post.username]['words'])
            no_posts_by_author = len(author_data[post.username]['posts'])

            for k in range(0, len(post.thread)):
                initial_post_id = post.thread[k]
                initial_preprocessed_index = str(discussion.discussion_id) + '-' + str(initial_post_id)
                initial_preprocessed = preprocessed_messages[initial_preprocessed_index]

                alignments = []
                for lemma in response_preprocessed:
                    if lemma in initial_preprocessed:
                        a_1 = 1
                    else:
                        a_1 = 0

                    a_2 = word_response_counter[lemma] / response_length

                    no_responses_with_word = len([post for post in author_data[post.username]['posts'] if lemma in post])

                    b_1 = no_responses_with_word / no_posts_by_author

                    b_2 = word_author_counter[lemma] / no_words_by_author

                    a = np.array([a_1, a_2])
                    b = np.array([b_1, b_2])

                    scaled_SCP_w = np.dot(a, weight) - np.dot(b, weight)
                    alignments.append(scaled_SCP_w)
                average_alignment = np.mean(alignments)
                distance = len(post.thread) - k
                data.append([
                    discussion.discussion_id,
                    initial_post_id,
                    post.post_id,
                    distance,
                    average_alignment
                ])
    print_i('task completed')

    print_t('storing alignment data')
    df = pd.DataFrame(data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'lexical_word_alignment'])
    df.to_csv(path)
    print_i('task completed')

    return df


"""
Generates a histogram for all discussions individually, turns per alignment bin
One histogram per 5 discussions
"""
def get_histograms_lexical_word_alignment_per_5(df):
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


"""
Generates one histogram for all discussions individually, turns per alignment bin
"""
def get_overall_histogram_lexical_word_alignment(df):
    discussion_ids = list(df['discussion_id'].unique())

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 300000) #300000 for linear
    # ax.set_yscale('log')

    ax.set_title('Lexical word alignment for all discussions')
    ax.set_xlabel('Alignment as Jaccard overlap')
    ax.set_ylabel('# of posts')

    # alignment is in range [0, 0.5], normalize to [0, 1]
    df['lexical_word_alignment'] = (df['lexical_word_alignment'] + 1) / 2

    for d_idx in discussion_ids:
        discussion_df = df[df['discussion_id'] == d_idx]
        alignment = discussion_df['lexical_word_alignment']
        ax.hist(alignment, bins=np.arange(0, 1, 0.025), alpha=0.1, label=str(d_idx)) #color='#d74a94'  histtype='step'
        print(d_idx)
    # sns.displot(df, x='lexical_word_alignment', hue='discussion_id', kde=True)

    fig.savefig('./Results/Lexical_word_alignment/all_histo_thread_stacked')
    fig.show()


"""
Generates one stacked histogram for all discussions combined, turns per alignment bin
"""
def get_overall_histogram_lexical_word_alignment_stacked(df, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100000) #2500000 for linear
    # ax.set_yscale('log')

    ax.set_title('Lexical word alignment for all discussions')
    ax.set_xlabel('Alignment as Jaccard overlap')
    ax.set_ylabel('# of posts')

    # alignment is in range [-1, 1], normalize to [0, 1]
    df['lexical_word_alignment'] = (df['lexical_word_alignment'] + 1) / 2

    alignment = df['lexical_word_alignment']
    ax.hist(alignment, bins=np.arange(0, 1, 0.025), alpha=0.8, color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(path)
    fig.show()

# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i'], ['e', 'b', 'c', 'd', 'j', 'g', 'h', 'i'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], ['e', 'f', 'a'])




