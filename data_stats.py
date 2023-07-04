#%% Imports

from Helpers import read_csv, print_t, print_i
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from linguistic_alignment_analysis.compute_lexical_word_alignment import jaccard_overlap, \
    preprocess_message_lexical_word, adapted_LLA
from linguistic_alignment_analysis.preprocessing_all import run_preprocessing, get_discusssion_posts, \
    remove_empty_discussions, replace_urls, get_discussion_threads, merge_consecutive_messages, \
    get_discussion_linear_threads
import copy
from collections import Counter
import pickle



# __datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_two_posts.csv' #use the unfiltered csv for original data
__datapath__ = './Data/discussion_post_text_date_author_parents_unfiltered.csv'
__datapath__filtered = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'
__discussion_length_histo_path__ = './Results/DataStats/discussion_length_histo.png'
__preprocessed_messages_linear__ = './Results/DataStats/TempStorage/pickle_preprocessed_linear'
__preprocessed_messages_thread__ = './Results/DataStats/TempStorage/pickle_preprocessed_thread'
__alignment_linear__ = './Results/DataStats/TempStorage/df_alignment_linear' # Where currently adapted LLA is stored
__alignment_linear_Jaccard__ = './Results/DataStats/TempStorage/df_alignment_linear_jaccard'
__alignment_linear_SCP__ = './Results/DataStats/TempStorage/df_alignment_linear_SCP'
__alignment_thread__ = './Results/DataStats/TempStorage/df_alignment_thread'
__discussion_alignment_histo_linear__ = './Results/DataStats/discussion_alignment_histo_linear.png'
__discussion_alignment_histo_thread__ = './Results/DataStats/discussion_alignment_histo_thread.png'
__avg_alignment_linear__ = './Results/DataStats/TempStorage/df_avg_alignment_linear.csv'
__avg_alignment_linear_LLA__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_LLA.csv'
__avg_alignment_linear_Jaccard__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_Jaccard.csv'
__avg_alignment_linear_SCP__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_SCP.csv'
__avg_alignment_thread__ = './Results/DataStats/TempStorage/df_avg_alignment_thread.csv'
__max_thread_linear__ = './Results/DataStats/TempStorage/df_max_thread_linear.csv'
__max_thread_thread__ = './Results/DataStats/TempStorage/df_max_thread_thread.csv'
__max_thread_histo_linear__ = './Results/DataStats/histo_max_thread_linear.png'
__max_thread_histo_thread__ = './Results/DataStats/histo_max_thread_thread.png'
__author_histo__ = './Results/DataStats/histo_author.png'
# __pickle_path_preprocessed_lexical_alignment__ = './PickleData/preprocessed_lexical_alignment_time_based_linear'
__pickle_path_df_lexical_word_preprocessed_linear__ = './PickleData/preprocessed_df_lexical_word_linear'



#%%


def get_message_length_stats():
    """
    Get message statistics
    """
    discussions = read_csv(__datapath__)

    length_of_the_messages = discussions["text"].str.split("\\s+")
    print('Average length of messages:', length_of_the_messages.str.len().mean())
    print('Min number of words: ', length_of_the_messages.str.len().min())
    print('Max number of words: ', length_of_the_messages.str.len().max())

    messages_more_than_512 = length_of_the_messages[length_of_the_messages.str.len() > 512]
    print(messages_more_than_512)
    print('Amount of posts larger than 512 words: ', len(messages_more_than_512.index))


def get_discussion_length_stats():
    """
    Gets discussion length statistics (with histogram for distributions and percentile)
    """
    discussions = read_csv(__datapath__filtered)

    discussion_list = []
    discussion_indices = discussions['discussion_id'].unique()
    for i in discussion_indices:
        discussion_df = discussions.loc[discussions['discussion_id'] == i]
        discussion_length = len(discussion_df)
        discussion_list.append([
            i,
            discussion_length
        ])

    discussion_df = pd.DataFrame(discussion_list, columns=['discussion_id', 'discussion_length'])
    print('mean: \t', discussion_df['discussion_length'].mean())
    print('median: \t', discussion_df['discussion_length'].median())
    print('min: \t', discussion_df['discussion_length'].min())
    print('max: \t', discussion_df['discussion_length'].max())
    print('percentiles: \t', discussion_df['discussion_length'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Length of discussions')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(0, 1300)
    ax1.set_ylim(0, 1100)
    ax2.set_xlim(0, 1300)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(8, 20)
    ax3.set_ylim(0, 250)
    ax3.set_xlabel('# posts')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(discussion_df['discussion_length'], bins=np.arange(8, 1300, 10), color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(discussion_df['discussion_length'], bins=np.arange(8, 1300, 10), color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(discussion_df['discussion_length'], bins=np.arange(8, 21, 1), color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(__discussion_length_histo_path__)
    fig.show()

    # discussion_99 = discussion_df[discussion_df['discussion_length'] > discussion_df['discussion_length'].quantile(.99)]
    # discussion_995 = discussion_df[discussion_df['discussion_length'] > discussion_df['discussion_length'].quantile(.995)]
    #
    # print('discussions larger than 99th percentile: ', discussion_99.to_string())
    # print('discussions larger than 995th percentile: ', discussion_995.to_string())


def get_preprocessed_overlap(discussions, storage_path):
    """
    Gets preprocessed messages to compute statistics of overlap
    :param discussions:
    :param storage_path:
    :return: preprocessed messages, df with for each message a list of lemmas.
    """
    print_t('preprocessing messages')
    preprocessed_posts = {}
    # get all the preprocessed posts
    data = []
    for i in discussions.keys():
        print('preprocessing', i)
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
    df.to_pickle(storage_path)
    print_i('task completed')

    return preprocessed_posts


def compute_lexical_word_alignment(discussions, preprocessed_messages, path):
    """
    Computes jaccard alignment between messages in discussions
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
                alignment = jaccard_overlap(initial_preprocessed, response_preprocessed)
                # alignment = adapted_LLA(initial_preprocessed, response_preprocessed)
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


def compute_lexical_word_alignment_SCP(discussions, preprocessed_messages, path, preprocessed_df_path):
    """
    Computes the actual alignment for each message and each of it's parent messages
    :param discussions: list of discussion objects
    :param preprocessed_messages: object of preprocessed messages
    :param path: path where to store the alignment dataframe
    :param preprocessed_df_path: path where preprocessed df is stored
    :return: dataframe with alignment
    """
    print_t('computing lexical word alignment for all messages and all of their parents')

    df = pd.read_pickle(preprocessed_df_path)

    print_t('Getting data per author')
    unique_authors = df['author_id'].unique()
    author_data = {}
    for author in unique_authors:
        df_author = df[df['author_id'] == author]
        posts = []
        preprocessed = []
        for index, row in df_author.iterrows():
            posts.append(row['preprocessed_text'])
            preprocessed = preprocessed + row['preprocessed_text']
        author_data[author] = {
            'posts': posts,
            'words': preprocessed
        }
    print_i('Task completed')

    word_set = set()
    for i in discussions.keys():
        discussion = discussions[i]
        print('discussion: ', i)
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index]
            word_set.update(response_preprocessed)
    unique_words = pd.Series(list(word_set))

    # proportion of words in R that are w
    # proportion of responses R by author i containing w
    # proportion of words by author i that are m

    print_t('initializing author lists and vars')
    # Amount of posts
    post_series = pd.Series(preprocessed_messages.keys())

    # Amount of authors
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

    # compute the fractions
    b1 = b1 / b1_posts
    b2 = b2 / b2_words

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

    print_i('computed alignment')

    print_t('storing alignment data')
    df = pd.DataFrame(alignment_data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'lexical_word_alignment'])
    df.to_csv(path)
    print_i('task completed')

    return df




def get_alignment_stats(discussion_df, storage_path):
    """
    Gets overlap statistics, histograms and percentile data
    :param discussion_df:
    :param storage_path:
    """


    # """
    print('mean: \t', discussion_df['average_alignment'].mean())
    print('median: \t', discussion_df['average_alignment'].median())
    print('min: \t', discussion_df['average_alignment'].min())
    print('max: \t', discussion_df['average_alignment'].max())
    print('percentiles: \t', discussion_df['average_alignment'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995, 1]))

    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig, (ax1, ax2) = plt.subplots(2)

    fig.suptitle('Average overlap between posts')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(-1, 1)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(-1, 1)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    # ax3.set_xlim(0.3, 0.65)
    # ax3.set_ylim(0, 500)
    ax2.set_xlabel('Adapted LLA')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(discussion_df['average_alignment'], bins=np.arange(-1, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(discussion_df['average_alignment'], bins=np.arange(-1, 1, 0.01),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    # ax3.hist(discussion_df['average_alignment'], bins=np.arange(0.3, 0.65, 0.005),
    #          color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(storage_path)
    fig.show()

    # """

    # """
    # discussion_low = discussion_df[discussion_df['average_alignment'] == 0]
    # sample_discussion_low = discussion_low.sample(n=10, random_state=1)
    # print('Random sample of discussions with low alignment', sample_discussion_low.to_string())

    # discussion_above_low = discussion_df[(discussion_df['average_alignment'] > discussion_df['average_alignment'].quantile(.005)) & (discussion_df['average_alignment'] < discussion_df['average_alignment'].quantile(.01))]
    # sample_discussion_above_low = discussion_above_low.sample(n=10, random_state=1)
    # print('Random sample of discussions with alignment a little higher: ', sample_discussion_above_low.to_string())
    
    # discussion_995 = discussion_df[discussion_df['average_alignment'] > discussion_df['average_alignment'].quantile(.995)]
    # sample_discussion_995 = discussion_995.sample(n=10, random_state=1)
    # print('Random sample of discussions with alignment in last 0.05th percentile: ', sample_discussion_995.to_string())

    # discussion_50 = discussion_df[(discussion_df['average_alignment'] < discussion_df['average_alignment'].quantile(.51)) & (discussion_df['average_alignment'] > discussion_df['average_alignment'].quantile(.49))]
    # sample_discussion_50 = discussion_50.sample(n=10, random_state=1)
    # print('Random sample of discussions with alignment at median:', sample_discussion_50.to_string())
    
    discussion_spikes = discussion_df[discussion_df['average_alignment'] > 0.4]
    print('amount of discussions in spikes: ', len(discussion_spikes))
    print('alignment spikes: ', discussion_spikes.to_string())
    
    # """

    print(len(discussion_df['discussion_id'].unique()))

    data = read_csv(__datapath__filtered)
    print(len(data['discussion_id'].unique()))



def get_average(discussions_df, path):
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
        # print(discussion_df)
        discussion_alignment_avg = discussion_df['lexical_word_alignment'].mean()
        averages.append([
            i,
            discussion_alignment_avg
        ])

    print(averages)
    average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_alignment'])
    average_df.to_csv(path)
    print(average_df)
    return average_df


def run_preprocessing_for_overlap_stats(datapath):
    """
    Running the preprocessing on a csv containing all discussions, posts, date, author, parent post etc for stats, without outliers removed.
    :param datapath: path where the csv data is read from
    :return: two lists of discussions as objects, for threads and for linear.
    """
    data = read_csv(datapath)
    unique_discussions = data['discussion_id'].unique().tolist()
    unique_discussions_shorter = unique_discussions[:50]
    data_shorter = data[data['discussion_id'].isin(unique_discussions_shorter)]
    discussion_posts = get_discusssion_posts(data_shorter)

    # discussion_posts = get_discusssion_posts(data)
    removed_empty = remove_empty_discussions(discussion_posts)
    replaced_urls = replace_urls(removed_empty)

    linear = get_discussion_linear_threads(copy.deepcopy(replaced_urls))
    linear_consecutive_merged = merge_consecutive_messages(linear)
    return linear_consecutive_merged



def get_overlap_stats():
    """
    Main function for getting the discussion data and redirecting to obtain overlap statistics
    """

    # linear_discussions = run_preprocessing_for_overlap_stats(__datapath__filtered)
    # preprocessed_linear = get_preprocessed_overlap(linear_discussions, __preprocessed_messages_linear__)
    # alignment_linear = compute_lexical_word_alignment_SCP(linear_discussions, preprocessed_linear, __alignment_linear_SCP__, __preprocessed_messages_linear__) # linear

    # load alignment data
    # alignment_thread = read_csv(__alignment_thread__)
    # alignment_linear = read_csv(__alignment_linear__)
    #
    # average_linear = get_average(alignment_linear, __avg_alignment_linear_SCP__)

    # load avg alignment data
    # average_thread = read_csv(__avg_alignment_thread__)
    average_linear = read_csv(__avg_alignment_linear_SCP__)

    print('linear data:')
    get_alignment_stats(average_linear, __discussion_alignment_histo_linear__)


def get_author_stats():
    """
    Obtains author statistics, histogram
    """
    discussions = read_csv(__datapath__filtered)

    print('# unique authors', len(discussions['author_id'].unique()))

    auth_list = []
    discussion_indices = discussions['discussion_id'].unique()
    for i in discussion_indices:
        print('Getting data', i)
        discussion_df = discussions.loc[discussions['discussion_id'] == i]
        no_authors = len(discussion_df['author_id'].unique())

        auth_list.append([
            i,
            no_authors
        ])

    author_discussion_df = pd.DataFrame(auth_list, columns=['discussion_id', 'no_authors'])
    print(author_discussion_df)

    print('mean: \t', author_discussion_df['no_authors'].mean())
    print('median: \t', author_discussion_df['no_authors'].median())
    print('min: \t', author_discussion_df['no_authors'].min())
    print('max: \t', author_discussion_df['no_authors'].max())
    print('percentiles: \t', author_discussion_df['no_authors'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995, 1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Number of authors in discussions')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(2, 160)
    # ax1.set_ylim(0, 5000)
    ax2.set_xlim(2, 160)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(2, 20)
    # ax3.set_ylim(0, 500)
    ax3.set_xlabel('# authors')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(author_discussion_df['no_authors'], bins=np.arange(2, 160, 2),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(author_discussion_df['no_authors'], bins=np.arange(2, 160, 2),
             color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(author_discussion_df['no_authors'], bins=np.arange(2, 21, 1),
             color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(__author_histo__)
    fig.show()




def get_author_contri_stats():
    """
    Obtains author contribution statistics, heatmap
    """

    discussions = read_csv(__datapath__filtered)

    print('# unique discussion', len(discussions['discussion_id'].unique()))
    print('# unique authors', len(discussions['author_id'].unique()))

    len_list = []
    discussion_indices = discussions['discussion_id'].unique()
    auth_post_list = []
    for i in discussion_indices:
        print('Getting data', i)
        discussion_df = discussions.loc[discussions['discussion_id'] == i]
        discussion_length = len(discussion_df)
        no_authors = len(discussion_df['author_id'].unique())

        len_list.append([
            i,
            discussion_length,
            no_authors
        ])

        author_indices = discussion_df['author_id'].unique()
        for a in author_indices:
            author_df = discussion_df.loc[discussions['author_id'] == a]
            no_posts = len(author_df)
            contribution = no_posts / discussion_length
            auth_post_list.append([
                i,
                a,
                no_posts,
                contribution
            ])

    len_discussion_df = pd.DataFrame(len_list, columns=['discussion_id', 'discussion_length', 'no_authors'])
    auth_post_df = pd.DataFrame(auth_post_list, columns=['discussion_id', 'author_id', 'no_posts', 'contribution'])

    print('mean: \t', auth_post_df['no_posts'].mean())
    print('median: \t', auth_post_df['no_posts'].median())
    print('min: \t', auth_post_df['no_posts'].min())
    print('max: \t', auth_post_df['no_posts'].max())
    print('percentiles: \t', auth_post_df['no_posts'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995, 1]))

    # x = []
    # y = []
    # z = []
    discussion_length_indices = len_discussion_df['discussion_length'].unique()
    max_discussion_length = len_discussion_df['discussion_length'].max()
    author_indices = len_discussion_df['no_authors'].unique()
    max_no_authors = len_discussion_df['no_authors'].max()
    empty_df = pd.DataFrame(np.nan, range(0, max_no_authors + 1), range(0, max_discussion_length + 1))

    for i in discussion_length_indices:
        print('constructing dataframe', i, 'out of', max_discussion_length)
        same_length_rows = len_discussion_df.loc[len_discussion_df['discussion_length'] == i]
        unique_no_authors = same_length_rows['no_authors'].unique()
        for j in unique_no_authors:
            no_discussions_length = len_discussion_df.loc[(len_discussion_df['discussion_length'] == i) & (len_discussion_df['no_authors'] == j)]
            empty_df.at[j, i] = len(no_discussions_length['discussion_id'].unique())

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    print('plotting graphs')
    # data_df_pivot = empty_df.pivot(columns="discussion_length", index="no_authors")

    norm_factor = empty_df.sum()
    data_df_pivot_normed = empty_df / norm_factor

    # fig = plt.figure(dpi=300, figsize=(9.5, 2.5))
    cs1 = ax1.pcolormesh(empty_df.values, norm="log", cmap="jet")
    cs2 = ax2.pcolormesh(data_df_pivot_normed.values, norm="log", cmap="jet")
    cs3 = ax3.pcolormesh(data_df_pivot_normed.values, norm="log", cmap="jet")

    # fig.set_figheight(10)
    # ax.axis('equal')
    fig.colorbar(cs1, ax=ax1, shrink=0.4, aspect=5)
    fig.colorbar(cs2, ax=ax2, shrink=0.4, aspect=5)
    fig.colorbar(cs3, ax=ax3, shrink=0.4, aspect=5)

    ax1.set_aspect('equal', 'box')
    ax1.set_xlim((0, 1300))
    ax1.set_ylim((0, 160))

    ax2.set_aspect('equal', 'box')
    ax2.set_ylabel('# of authors')
    ax2.set_xlim((0, 1300))
    ax2.set_ylim((0, 160))

    ax3.set_aspect('equal', 'box')
    ax3.set_xlabel('Discussion length')
    ax3.set_xlim((0, 325))
    ax3.set_ylim((0, 40))
    # ax.set(xlim=(0, 1300), ylim=(0, 160))

    plt.suptitle('Author contribution')
    plt.tight_layout()
    plt.savefig('Results/DataStats/AuthorStats/author_contribution_heat_combined_discussions.png')
    # plt.show()

    """
    # plt.xlim(1300)
    # plt.ylim(160)
    cs = plt.pcolormesh(data_df_pivot.values, cmap='jet', norm='log')
    bar = plt.colorbar(cs)
    # tick_levels = [0, 10, 10**2, 10**3, 10**4, 10**5]
    # bar.ax.set_yticks(tick_levels)
    plt.suptitle('Contribution of authors')
    plt.show()

    norm_factor = data_df_pivot.sum()
    data_df_pivot_normed = data_df_pivot / norm_factor
    plt.pcolormesh(data_df_pivot_normed.values, cmap='jet', norm='log')
    plt.show()
    """

    no_most_authors = 10
    auth_post_df_d_indices = auth_post_df['discussion_id'].unique()
    bar_data = pd.DataFrame(0, auth_post_df_d_indices, range(0, no_most_authors + 1))


    for d_idx in auth_post_df_d_indices:
        print('idx', d_idx)
        d_df = auth_post_df[auth_post_df['discussion_id'] == d_idx]
        d_df = d_df.sort_values(by='contribution', ascending=False)
        first_10 = d_df.head(no_most_authors)
        first_10 = first_10.reset_index()
        for i in first_10.index:
            bar_data[i][d_idx] = first_10['contribution'][i]

        if len(d_df.index) > no_most_authors:
            rest = d_df.tail(-no_most_authors)
            rest_contribution = rest['contribution'].sum()
            bar_data[no_most_authors][d_idx] = rest_contribution

    counts = np.zeros(no_most_authors + 1)
    error = np.zeros((2, no_most_authors + 1))
    for i in range(0, no_most_authors + 1):
        median = bar_data[i].median()
        counts[i] = median
        error[0][i] = median - bar_data[i].min()
        error[1][i] = bar_data[i].max() - median
        if error[1][i] > 0.8:
            print(bar_data[i].max())
            print(bar_data[i].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995]))

    fig, ax = plt.subplots()

    authors = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', 'rest']
    # bar_labels = ['red', 'blue', '_red', 'orange']
    # bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.bar(authors, counts, yerr=error, color='#d74a94')

    ax.set_xlabel('Most prolific authors')
    ax.set_ylabel('Median fraction of contribution')
    ax.set_ylim((0, 1))
    print('Mean contributions', counts)
    ax.set_title('Fraction of contribution for most prolific authors')
    plt.savefig('Results/DataStats/AuthorStats/author_contribution_bar_10_median.png')

    plt.show()


def get_max_threads(discussions):
    """
    Returns the longest thread (including last message) in a discussion
    :param discussions: list of discussion objects
    :return: df with longest thread per discussion
    """

    max_threads = []
    for d_idx in discussions.keys():
        discussion = discussions[d_idx]
        max_thread = []
        for p_idx in discussion.posts.keys():
            post_thread = discussion.posts[p_idx].thread
            post_thread_appended = copy.deepcopy(post_thread)
            post_thread_appended.append(discussion.posts[p_idx].post_id)
            if len(post_thread_appended) > len(max_thread):
                max_thread = post_thread_appended
        max_threads.append([
            d_idx,
            max_thread,
            len(max_thread)
        ])

    max_thread_df = pd.DataFrame(max_threads, columns=['discussion_id', 'max_thread', 'len_max_thread'])
    return max_thread_df


def get_pretty_max_thread_stats(discussion_df, storage_path):
    """
    Prints the max thread stats: mean, avg, percentiles and histogram
    :param discussion_df: dataframe with max thread data per discussion
    :param storage_path: path where to store the histogram
    """
    print('mean: \t', discussion_df['len_max_thread'].mean())
    print('median: \t', discussion_df['len_max_thread'].median())
    print('min: \t', discussion_df['len_max_thread'].min())
    print('max: \t', discussion_df['len_max_thread'].max())
    print('percentiles: \t', discussion_df['len_max_thread'].describe(percentiles=[.01, .05, .1, .9, .95, .99, .995]))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Length of longest threads')
    fig.subplots_adjust(hspace=0.5)

    ax1.set_xlim(0, 1300)
    ax1.set_ylim(0, 10000)
    ax2.set_xlim(0, 1300)
    ax2.set_yscale('log')
    ax2.set_ylabel('# discussions')
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 1000)
    ax3.set_xlabel('# posts')
    # ax.set_ylim(0, 100000) #2500000 for linear

    # for ax in fig.get_axes():
    #     ax.set_xlabel('# posts')
    #     ax.set_ylabel('# discussions')

    ax1.hist(discussion_df['len_max_thread'], bins=np.arange(0, 1300, 10), color='#d74a94')  # color='#d74a94'  histtype='step'
    ax2.hist(discussion_df['len_max_thread'], bins=np.arange(0, 1300, 10), color='#d74a94')  # color='#d74a94'  histtype='step'
    ax3.hist(discussion_df['len_max_thread'], bins=np.arange(0, 1300, 1), color='#d74a94')  # color='#d74a94'  histtype='step'

    fig.savefig(storage_path)
    fig.show()


def get_max_thread_stats():
    threads_discussions, linear_discussions = run_preprocessing(__datapath__)
    # threads_discussions = {1: threads_discussions[1]}
    # linear_discussions = {1: linear_discussions[1]}
    print('getting threaded max')
    max_thread_df_thread = get_max_threads(threads_discussions)
    print(max_thread_df_thread)
    print('getting linear max')
    max_thread_df_linear = get_max_threads(linear_discussions)
    print(max_thread_df_linear)
    max_thread_df_thread.to_csv(__max_thread_thread__)
    max_thread_df_linear.to_csv(__max_thread_linear__)
    get_pretty_max_thread_stats(max_thread_df_thread, __max_thread_histo_thread__)
    get_pretty_max_thread_stats(max_thread_df_linear, __max_thread_histo_linear__)



# get_message_length_stats()
# get_discussion_length_stats()
# get_overlap_stats()
# get_author_stats()
get_author_contri_stats()
# get_max_thread_stats()


#%% Find word stats
# Reset discussions (just necessary for python console)
discussions_preprocessed = {}

#Load preprocessed data (see preprocessing.py)

print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_df_lexical_word_preprocessed_linear__))
store_file = open(__pickle_path_df_lexical_word_preprocessed_linear__, 'rb')
discussions_preprocessed = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%%
discussion_series = discussions_preprocessed['preprocessed_text']
corpus_counter = Counter()
for index, row in discussion_series.items():
    post_counter = Counter(row)
    corpus_counter.update(post_counter)


#%%
labels, values = zip(*corpus_counter.items())

indexes = np.arange(len(labels))
width = 1

plt.hist(values)
plt.show()