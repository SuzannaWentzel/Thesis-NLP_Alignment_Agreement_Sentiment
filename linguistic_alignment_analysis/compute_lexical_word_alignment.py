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
import seaborn as sns

from Helpers import color_scheme

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
def preprocess_message_LILLA(post):
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


"""
Preprocesses each message for this analysis
"""
def get_preprocessed_messages_for_lexical_word(discussions):
    print('[TASK] preprocessing messages')
    preprocessed_posts = {}
    # get all the preprocessed posts
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_LILLA(post.message)
            preprocessed_posts[str(i) + '-' + str(j)] = preprocessed
    print('[INFO] task completed')
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
Computes the actual alignment for each message and each of it's parent messages
"""
def compute_lexical_word_alignment(discussions, preprocessed_messages, path):
    print('[TASK] computing lexical word alignment for all messages and all of their parents')
    data = []
    for i in discussions.keys():
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
    print('[INFO] task completed')

    print('[TASK] storing alignment data')
    df = pd.DataFrame(data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'lexical_word_alignment'])
    df.to_csv(path)
    print('[INFO] task completed')

    return discussions


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
        print(plot_disc_ids)

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
    df['lexical_word_alignment'] = df['lexical_word_alignment'] * 2

    for d_idx in discussion_ids:
        discussion_df = df[df['discussion_id'] == d_idx]
        alignment = discussion_df['lexical_word_alignment']
        ax.hist(alignment, bins=np.arange(0, 1, 0.025), alpha=0.1, label=str(d_idx)) #color='#d74a94'  histtype='step'
        print(d_idx)
    # sns.displot(df, x='lexical_word_alignment', hue='discussion_id', kde=True)

    fig.savefig('./Results/Lexical_word_alignment/all_histo_linear')
    fig.show()

# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd', 'e', 'b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i'], ['e', 'b', 'c', 'd', 'j', 'g', 'h', 'i'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['b', 'c', 'd'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd'], ['e', 'f', 'a'])
# jaccard_overlap(['a', 'b', 'c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], ['e', 'f', 'a'])




