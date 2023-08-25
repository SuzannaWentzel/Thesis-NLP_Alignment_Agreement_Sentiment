#%% imports
from Helpers import print_t, print_i, store_data
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import string
import math


# __preprocessing_adapted_LLA_avg_linear__ = 'AlignmentData/preprocessing_LLA_alignment_linear.csv'
__pickle_path_preprocessed__ = './PickleData/preprocessed_time_based_linear'
__pickle_path_df_lexical_word_preprocessed_linear__ = './PickleData/preprocessed_df_lexical_word_linear'
__pickle_path_preprocessed_lexical_alignment__ = './PickleData/preprocessed_lexical_alignment_time_based_linear'
__csv_alignment_data__ = './AlignmentData/lexical_alignment_all.csv'
__pickle_path_best_alignment_clustering_data__ = './PickleData/best_alignment_clustering_data'
__pickle_path_bin_ids__ = './PickleData/bin_ids'
__csv_alignment_class_data__ = './AlignmentData/alignment_classes.csv'
__csv_alignment_class_delta_data__ = './AlignmentData/alignment_classes_deltas.csv'



#%% Load preprocessed data (see preprocessing.py)
# Reset discussions (just necessary for python console)
discussions = {}
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_preprocessed__))
store_file = open(__pickle_path_preprocessed__, 'rb')
discussions = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Load preprocessing functions for lexical word
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
    # post_punctuation_stopwords_removed = [token.lower() for token in tokens if token not in string.punctuation and token not in stop_words]
    post_stopwords_removed = [token.lower() for token in tokens if token not in stop_words]
    # Apply POS
    tagged = nltk.pos_tag(post_stopwords_removed)
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


#%% Preprocess messages for lexical alignment
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
df.to_pickle(__pickle_path_df_lexical_word_preprocessed_linear__)
print_i('stored preprocessing data')


#%% Load data
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_df_lexical_word_preprocessed_linear__))
store_file = open(__pickle_path_df_lexical_word_preprocessed_linear__, 'rb')
df = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Removing empty messages (when only consisted of exclamations or stopwords or a combination
for i in discussions.keys():
    discussion = discussions[i]
    to_remove_posts = []
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        preprocessed_message = preprocessed_posts[str(i) + '-' + str(j)]
        if len(preprocessed_message) == 0:
            print_i(f'Found post to remove in discussion_id {i} and post_id {j} ({post.post_id})')
            to_remove_posts.append(post.post_id)

    # Remove the posts
    for j in to_remove_posts:
        del discussion.posts[j]

    # update threads to not include the removed messages
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        new_thread = [post_id for post_id in post.thread if post_id not in to_remove_posts]
        post.set_thread(new_thread)


#%% Store discussions preprocessed for lexical alignment analysis
store_data(discussions, __pickle_path_preprocessed_lexical_alignment__)

#%% Load prerprocessed discussions
discussions = {}
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_preprocessed_lexical_alignment__))
store_file = open(__pickle_path_preprocessed_lexical_alignment__, 'rb')
discussions = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')

#%% Remove discussions that have now less than two authors with at least 4 posts
# discussions_to_remove = []
# for i in discussions.keys():
#     author_list = {}
#     discussion = discussions[i]
#     for j in discussion.posts.keys():
#         post = discussion.posts[j]
#         if not author_list[post.username]:
#             author_list[post.username] = 1
#         else:
#             author_list[post.username] += 1
#
#     authors_enough = 0
#     for a in author_list.keys():
#         if author_list[a] >= 4:
#             authors_enough += 1
#
#     if authors_enough < 2:
#         discussions_to_remove.append(i)
#
# print_i(f'Discussions to remove: {discussions_to_remove}')

# for i in discussions_to_remove:
#     del discussions[i]

#%% Compute alignment
print_t('computing lexical word alignment for all messages and all of their parents')
data_alignment = []
for i in discussions.keys():
    print('computing alignment', i)
    discussion = discussions[i]
    discussion_df = df.loc[df['discussion_id'] == i]
    # print('1', discussion_df)

    unique_authors = df['author_id'].unique()
    vocab = {a: [] for a in unique_authors}
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
        response_preprocessed = preprocessed_posts[response_preprocessed_index]
        vocab_not_author = vocab[post.username]
        if len(vocab_not_author) != 0:
            tokens_repeated = [word for word in response_preprocessed if word in vocab_not_author]
            token_repetition = 0
            if len(response_preprocessed) > 0:
                token_repetition = len(tokens_repeated) / len(response_preprocessed)
            else:
                print('Found 0 at discussion', discussion.discussion_id)
                print('At post: ', post.post_id)

            data_alignment.append([
                discussion.discussion_id,
                post.post_id,
                token_repetition
            ])

        for author in unique_authors:
            if author != post.username:
                vocab[author] += response_preprocessed

        # vocab = {a: vocab_a + response_preprocessed for a, vocab_a in vocab if a != post.username}
        # vocab[post.username] = vocab_not_author

print('[TASK] storing alignment data')
alignment_df = pd.DataFrame(data_alignment, columns=['discussion_id', 'post_id', 'lexical_word_alignment'])
alignment_df.to_csv(__csv_alignment_data__)
print('[INFO] task completed')


#%% Load alignment data
alignment_df = pd.read_csv(__csv_alignment_data__)

#%% Plot data
discussion_length = []
unique_disc_idxs = alignment_df['discussion_id'].unique()

for d_idx in unique_disc_idxs:
    discussion = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    alignment_vals = discussion['lexical_word_alignment']
    p_idxs = range(1, len(alignment_vals) + 1)

    plt.plot(p_idxs, alignment_vals, linewidth=0.5)
    # plt.scatter(p_idxs, alignment_vals, color='#d74a94', s=1)

plt.xlabel('time (in posts)')
plt.xlim((0, 1103))
plt.ylim((0, 1))
plt.ylabel('Time-based word overlap')
plt.suptitle('Word overlap over time')
plt.savefig('Results/Lexical_word_alignment/Time/line_alignment_time_all.png')
plt.show()

#%% Get alignment distribution for all messages
print('mean: \t', alignment_df['lexical_word_alignment'].mean())
print('percentiles: \t', alignment_df['lexical_word_alignment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Time-based overlap in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# posts')
ax2.set_xlabel('Time-based overlap')

ax1.hist(alignment_df['lexical_word_alignment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(alignment_df['lexical_word_alignment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.savefig('Results/Lexical_word_alignment/histo_alignment_messages.png')
fig.show()


#%% Get average alignment
averages = []
unique_disc_idxs = alignment_df['discussion_id'].unique()

for d_idx in unique_disc_idxs:
    print('averaging alignment', d_idx)
    discussion_df = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    discussion_alignment_avg = discussion_df['lexical_word_alignment'].mean()
    averages.append([
        d_idx,
        discussion_alignment_avg
    ])

average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_alignment'])

#%% Plot alignment distribution
print('mean: \t', average_df['average_alignment'].mean())
print('percentiles: \t', average_df['average_alignment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Average time-based overlap in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# discussions')
ax2.set_xlabel('Mean time-based overlap')

ax1.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(average_df['average_alignment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.show()

discussion_spikes_low = average_df[average_df['average_alignment'] < 0.15]
print('amount of discussions in spikes in beginning: ', len(discussion_spikes_low))
print('alignment spikes: ', discussion_spikes_low.to_string())


discussion_1 = average_df[average_df['average_alignment'] < average_df['average_alignment'].quantile(.01)]
sample_discussion_1 = discussion_1.sample(n=10, random_state=1)
print('Random sample of discussions with alignment in first percentile: ', sample_discussion_1.to_string())

discussion_50 = average_df[(average_df['average_alignment'] < average_df['average_alignment'].quantile(.51)) & (average_df['average_alignment'] > average_df['average_alignment'].quantile(.49))]
sample_discussion_50 = discussion_50.sample(n=10, random_state=1)
print('Random sample of discussions with alignment at median:', sample_discussion_50.to_string())

discussion_995 = average_df[average_df['average_alignment'] > average_df['average_alignment'].quantile(.995)]
sample_discussion_995 = discussion_995.sample(n=10, random_state=1)
print('Random sample of discussions with alignment in last 0.05th percentile: ', sample_discussion_995.to_string())

discussion_spikes_high = average_df[average_df['average_alignment'] > 0.9]
print('amount of discussions in spikes on end: ', len(discussion_spikes_high))
print('alignment spikes: ', discussion_spikes_high.to_string())




#%% Obtain timeseries clustering
# add correct post index
df_times = alignment_df.copy()
for d_idx in df_times['discussion_id'].unique():
    discussion = df_times.loc[df_times['discussion_id'] == d_idx]
    df_times.loc[df_times['discussion_id'] == d_idx, 'time_post_id'] = range(0, len(discussion))

df_times = df_times.drop(columns='post_id')
pivoted = df_times.pivot(index='discussion_id', columns='time_post_id')


model = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10)
y = model.fit_predict(pivoted.values)
x = df_times['time_post_id'].unique()


#%% Plot time series
predicted_classes = pd.DataFrame(data=y, index=pivoted.index, columns=['class'])
unique_classes = predicted_classes['class'].unique()

fig, axs = plt.subplots(len(unique_classes))

for u_class in unique_classes:
    ax = axs[u_class]
    discussions_with_class = predicted_classes.loc[predicted_classes['class'] == u_class]
    discussion_ids_with_class = discussions_with_class.index
    discussions_df_with_class = df_times.loc[df_times['discussion_id'].isin(discussion_ids_with_class)]
    for d_idx in discussion_ids_with_class:
        discussion = df_times.loc[df_times['discussion_id'] == d_idx]
        ax.plot(discussion['time_post_id'], discussion['lexical_word_alignment'])

axs[int(np.median(unique_classes))].set_ylabel('Lexical word alignment')
axs[len(unique_classes)-1].set_xlabel('Posts in time')
fig.suptitle('Alignment over time per found class')
fig.show()


#%% Plot time series with trends
predicted_classes = pd.DataFrame(data=y, index=pivoted.index, columns=['class'])
unique_classes = predicted_classes['class'].unique()

fig, axs = plt.subplots(len(unique_classes))
fig.subplots_adjust(hspace=0.3)
fig.set_size_inches(8.5, 10)


for u_class in unique_classes:
    ax = axs[u_class]
    discussions_with_class = predicted_classes.loc[predicted_classes['class'] == u_class]
    discussion_ids_with_class = discussions_with_class.index
    discussions_df_with_class = df_times.loc[df_times['discussion_id'].isin(discussion_ids_with_class)]
    discussions_pivoted_df_with_class = pivoted.loc[discussion_ids_with_class]
    for d_idx in discussion_ids_with_class:
        discussion = df_times.loc[df_times['discussion_id'] == d_idx]
        ax.plot(discussion['time_post_id'], discussion['lexical_word_alignment'])
    trend = discussions_pivoted_df_with_class.mean()
    mean_per_post = trend.reset_index()
    ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
    # ax.set_xticks([])

axs[int(np.median(unique_classes))].set_ylabel('Time-based word repetition per class')
axs[len(unique_classes)-1].set_xlabel('Posts in time')

fig.suptitle('Alignment over time per found class')
fig.show()


#%% Cumsum rolling average
def rolling_average(a, n=3):
    ret = np.cumsum(a, dtype=float) # computes cumulative sum for
    ret[n:] = ret[n:] - ret[:-n] # subtract cumsum before window from cumsum at end of window to get the sum inside the window (bc of the shift)
    res = ret[n - 1:] / n # divide by windowsize, throw away values outside of possible windows
    return res


#%% Plot trends
discussion_length = []
unique_disc_idxs = alignment_df['discussion_id'].unique()


window_size = 5
data_rolling_average = []
for d_idx in unique_disc_idxs:
    discussion = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    alignment_vals = discussion['lexical_word_alignment']
    p_idxs = range(1 + (window_size//2), len(alignment_vals) + 1 - (window_size//2))
    rolling_averages = rolling_average(alignment_vals.values, n=window_size)
    plt.plot(p_idxs, rolling_averages, linewidth=0.5)
    data_rolling_average_disc = [[d_idx, p_idxs[i], rolling_averages[i]] for i in range(0, len(p_idxs))]
    data_rolling_average += data_rolling_average_disc

plt.xlabel('time (in posts)')
plt.xlim((0, 1103))
plt.ylim((0, 1))
plt.ylabel('Rolling average of time-based word overlap')
plt.suptitle('Rolling average of word overlap over time')
plt.savefig('Results/Lexical_word_alignment/Time/line_alignment_RA_time_all.png')
plt.show()

rolling_average_df = pd.DataFrame(data_rolling_average, columns=['discussion_id', 'time_post_id', 'rolling_average'])

#%% Apply clustering to rolling average
pivoted_rolling_average = rolling_average_df.pivot(index='discussion_id', columns='time_post_id')
model_rolling_avg = TimeSeriesKMeans(n_clusters=15, metric="dtw", max_iter=10)
y_ra = model_rolling_avg.fit_predict(pivoted_rolling_average.values)
x_ra = rolling_average_df['time_post_id'].unique()

#%% Plot clustered classes with rolling averages
predicted_classes_ra = pd.DataFrame(data=y_ra, index=pivoted_rolling_average.index, columns=['class'])
unique_classes_ra = predicted_classes_ra['class'].unique()

fig, axs = plt.subplots(len(unique_classes_ra))
fig.subplots_adjust(hspace=1)
fig.set_size_inches(8.5, 50)

for u_class in unique_classes_ra:
    ax = axs[u_class]
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1300))
    discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
    discussion_ids_with_class = discussions_with_class.index
    discussions_df_with_class = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(discussion_ids_with_class)]
    discussions_pivoted_df_with_class = pivoted_rolling_average.loc[discussion_ids_with_class]
    for d_idx in discussion_ids_with_class:
        discussion = rolling_average_df.loc[rolling_average_df['discussion_id'] == d_idx]
        ax.plot(discussion['time_post_id'], discussion['rolling_average'])
    trend = discussions_pivoted_df_with_class.mean()
    mean_per_post = trend.reset_index()
    ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
    # ax.set_xticks([])

axs[int(np.median(unique_classes_ra))].set_ylabel('Time-based word repetition per class with rolling averages')
axs[len(unique_classes_ra)-1].set_xlabel('Posts in time')

fig.suptitle('Alignment over time per found class')
fig.show()


#%% Get discussion length
discussion_length = []
for d_idx in unique_disc_idxs:
    discussion = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    discussion_length.append([
        d_idx,
        len(discussion)
    ])

length_df = pd.DataFrame(discussion_length, columns=['discussion_id', 'no_posts'])


#%% Get discussion length percentiles
percentiles = length_df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
print(percentiles)

#%% Get bins
count_discussions_per_length = length_df.groupby('no_posts').count().rename(columns={'discussion_id':'count'})
total_discussion_count = length_df.shape[0]

bin_count = 5
bin_threshold = total_discussion_count / bin_count

lengths_for_bins = []
lengths_for_bin = []
bin_sizes = []
bin_disc_count = 0
for i, row in count_discussions_per_length.iterrows():
    bin_disc_count += row['count']
    lengths_for_bin.append(i)

    if bin_disc_count >= bin_threshold:
        lengths_for_bins.append(lengths_for_bin)
        lengths_for_bin = []
        bin_sizes.append(bin_disc_count)
        bin_disc_count = 0

lengths_for_bins.append(lengths_for_bin)
bin_sizes.append(bin_disc_count)
del i, row, lengths_for_bin, bin_disc_count


#%% Last bin should be split into multiple, too large of a variance in discussion lengths
# last_bin = lengths_for_bins[-1]
# last_bin_start = int(last_bin[0])
# last_bin_threshold = 50
# bin_threshold = last_bin_start + last_bin_threshold
# del lengths_for_bins[-1]
#
# lengths_for_bin = []
# for i in last_bin:
#     if i >= bin_threshold and len(lengths_for_bin) >= last_bin_threshold:
#         lengths_for_bins.append([
#             lengths_for_bin
#         ])
#         lengths_for_bin = []
#         bin_threshold += last_bin_threshold
#
#     lengths_for_bin.append(i)

last_bin = lengths_for_bins[-1]
del lengths_for_bins[-1]

last_bins = [(87, 137), (137, 187), (187, 237), (237, 287)] # TODO: not hardcode
for bin in last_bins:
    lengths_for_bin = [i for i in range(bin[0], bin[1]) if i in last_bin]
    lengths_for_bins.append(lengths_for_bin)


# %% SOS for different k's for each bin

# Go through length bins
bin_counter = 0
for bin_lengths in lengths_for_bins:
    bin_counter += 1
    disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

    # for each bin, try out different ks and compute mean sum of squares
    inertias = []
    ks = [k for k in range(1, 11)]
    for n in ks:
        model_rolling_avg = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10)
        y_ra = model_rolling_avg.fit_predict(pivoted_discussions_in_bin_length.values)
        x_ra = discussions_in_bin_length['time_post_id'].unique()

        predicted_classes_ra = pd.DataFrame(data=y_ra, index=pivoted_discussions_in_bin_length.index, columns=['class'])
        unique_classes_ra = predicted_classes_ra['class'].unique()

        inertias.append(model_rolling_avg.inertia_)

        # Plot these classes to see how they are doing
        fig, axs = plt.subplots(len(unique_classes_ra), figsize=(5+(1 * (bin_lengths[-1] / 50)), 8*n))
        fig.tight_layout()
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        for u_class in unique_classes_ra:
            discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
            discussion_ids_with_class = discussions_with_class.index
            discussions_df_with_class = discussions_in_bin_length.loc[
                discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
            discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

            ax = axs
            if n > 1:
                ax = axs[u_class]
            ax.set_ylim((0, 1))
            ax.set_xlim((0, bin_lengths[-1]))
            for d_idx in discussion_ids_with_class:
                discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
                ax.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
            trend = discussions_pivoted_df_with_class.mean()
            mean_per_post = trend.reset_index()
            ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
            # ax.set_xticks([])

        if n > 1:
            axs[int(np.median(unique_classes_ra))].set_ylabel(
                'Time-based word repetition per class with rolling averages')
            axs[len(unique_classes_ra) - 1].set_xlabel('Posts in time')
        else:
            axs.set_ylabel(
                'Time-based word repetition per class with rolling averages')
            axs.set_xlabel('Posts in time')

        fig.suptitle(f'Alignment over time for bin {bin_counter}, k={n}')
        fig.savefig(f'Results/Clustering/5_bins_and_last_splitted/line_alignment_bin_{bin_counter}_k_{n}')
        fig.show()

    fig_elbow, ax_elbow = plt.subplots(figsize=(4, 4))
    fig_elbow.tight_layout()
    fig_elbow.subplots_adjust(top=0.9, left=0.2, right=0.95, bottom=0.15)
    ax_elbow.plot(ks, inertias, color='#d74a94')
    ax_elbow.set_xlabel('number of clusters (k)')
    ax_elbow.set_ylabel('inertia')
    fig_elbow.suptitle(f'Clustering for bin {bin_counter} (length {bin_lengths[0]} - {bin_lengths[-1]})')
    fig_elbow.savefig(f'Results/Clustering/5_bins_and_last_splitted/line_sos_bin_{bin_counter}')
    fig_elbow.show()


#%% Cluster per bin
ks_per_bin = [4, 6, 5, 5, 4, 5, 5, 4]
tries = 5
best_models = []
all_models = []

# Go through length bins
for i, bin_lengths in enumerate(lengths_for_bins):
    disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

    # for each bin, try out different ks and compute mean sum of squares
    inertias = []
    k = ks_per_bin[i]
    best_model = {}

    for n in range(0, tries):
        model_rolling_avg = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10)
        y_ra = model_rolling_avg.fit_predict(pivoted_discussions_in_bin_length.values)
        x_ra = discussions_in_bin_length['time_post_id'].unique()

        predicted_classes_ra = pd.DataFrame(data=y_ra, index=pivoted_discussions_in_bin_length.index, columns=['class'])
        unique_classes_ra = predicted_classes_ra['class'].unique()

        inertia = model_rolling_avg.inertia_
        inertias.append(inertia)
        all_models.append([
            i,
            n,
            model_rolling_avg,
            y_ra,
            predicted_classes_ra,
            inertia,
        ])

        # Keep this model if it performs better than a previously stored one
        if len(best_model.keys()) == 0:
            best_model['model'] = model_rolling_avg
            best_model['y_ra'] = y_ra
            best_model['predicted_classes_ra'] = predicted_classes_ra
            best_model['inertia'] = inertia
            best_model['try_counter'] = n
        elif inertia < best_model['inertia']:
            best_model['model'] = model_rolling_avg
            best_model['y_ra'] = y_ra
            best_model['predicted_classes_ra'] = predicted_classes_ra
            best_model['inertia'] = inertia
            best_model['try_counter'] = n

        # Plot his attempt and store to be able to compare later
        fig, axs = plt.subplots(math.ceil(len(unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_lengths[-1] / 50)), 4 * k))
        fig.tight_layout()
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        for i_class, u_class in enumerate(unique_classes_ra):
            discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
            discussion_ids_with_class = discussions_with_class.index
            discussions_df_with_class = discussions_in_bin_length.loc[
                discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
            discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

            ax_x = math.floor(i_class/2)
            ax_y = i_class % 2
            ax = axs[ax_x, ax_y]
            ax.set_ylim((0, 1))
            ax.set_xlim((0, bin_lengths[-1]))
            for d_idx in discussion_ids_with_class:
                discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
                ax.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
            trend = discussions_pivoted_df_with_class.mean()
            mean_per_post = trend.reset_index()
            ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
            # ax.set_xticks([])

        axs[math.floor((len(unique_classes_ra)-1)/2/2), 0].set_ylabel(
            'Time-based word repetition per class with rolling averages')
        axs[math.floor((len(unique_classes_ra)-1)/2), 0].set_xlabel('Posts in time')
        axs[math.floor((len(unique_classes_ra)-1)/2), 1].set_xlabel('Posts in time')

        fig.suptitle(f'Alignment over time for bin {i+1}, attempt {n+1}')
        fig.savefig(f'Results/Clustering/Tries/line_alignment_bin_{i+1}_attempt_{n+1}')
        fig.show()

    # store the best model
    best_models.append(best_model)

    # Plot and save the best clustering:
    best_model_y_ra = best_model['y_ra']
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
    best_n = best_model['try_counter']

    fig_best, axs_best = plt.subplots(math.ceil(len(best_model_unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_lengths[-1] / 50)), 4 * k))
    fig_best.tight_layout()
    fig_best.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    for i_class, u_class in enumerate(best_model_unique_classes_ra):
        discussions_with_class = best_model_predicted_classes_ra.loc[best_model_predicted_classes_ra['class'] == u_class]
        discussion_ids_with_class = discussions_with_class.index
        discussions_df_with_class = discussions_in_bin_length.loc[
            discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
        discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

        ax_x = math.floor(i_class/2)
        ax_y = i_class % 2
        ax = axs_best[ax_x, ax_y]
        ax.set_ylim((0, 1))
        ax.set_xlim((0, bin_lengths[-1]))
        for d_idx in discussion_ids_with_class:
            discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
            ax.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
        trend = discussions_pivoted_df_with_class.mean()
        mean_per_post = trend.reset_index()
        ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
        # ax.set_xticks([])

    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2/2), 0].set_ylabel(
        'Time-based word repetition per class with rolling averages')
    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2), 0].set_xlabel('Posts in time')
    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2), 1].set_xlabel('Posts in time')

    fig_best.suptitle(f'Alignment over time for bin {i+1} (lengths {bin_lengths[0]}-{bin_lengths[-1]})')
    fig_best.savefig(f'Results/Clustering/Best_Result/line_alignment_bin_{i+1}_attempt_{best_n+1}')
    fig_best.show()


#%% Store best found models
store_data(best_models, __pickle_path_best_alignment_clustering_data__ + '_backup')


#%% Load best found models
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_best_alignment_clustering_data__))
store_file = open(__pickle_path_best_alignment_clustering_data__, 'rb')
best_models = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')



#%% Get amount of discussions per bin
for i, bin_lengths in enumerate(lengths_for_bins):
    bin_counter += 1
    disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
    print_i(f'Bin {i + 1}: \t {len(disc_ids_with_length)} discussions')


#%% Get discussion ids per bin
bin_ids = []
for bin_lengths in lengths_for_bins:
    bin_ids.append(length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique())


#%% Store discussion ids per bin
store_data(bin_ids, __pickle_path_bin_ids__)


#%% Load bins that were used before
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_bin_ids__))
store_file = open(__pickle_path_bin_ids__, 'rb')
bin_ids = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Pretty print of pandas df function
def pretty_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


#%% Print words that are overlapped
test_discussion_index = 6687
test_discussion = discussions[test_discussion_index]
test_data_alignment = []
# for post in test_discussion.values():
test_discussion_df = df.loc[df['discussion_id'] == test_discussion_index]
test_unique_authors = test_discussion_df['author_id'].unique()
vocab = {a: [] for a in test_unique_authors}
tokens_repeated_per_post = {}
for j in test_discussion.posts.keys():
    post = test_discussion.posts[j]
    response_preprocessed_index = str(test_discussion.discussion_id) + '-' + str(post.post_id)
    response_preprocessed = preprocessed_posts[response_preprocessed_index]
    vocab_not_author = vocab[post.username]
    if len(vocab_not_author) != 0:
        tokens_repeated = [word for word in response_preprocessed if word in vocab_not_author]
        tokens_repeated_per_post[post.post_id] = tokens_repeated
        token_repetition = 0
        if len(response_preprocessed) > 0:
            token_repetition = len(tokens_repeated) / len(response_preprocessed)
        else:
            print('Found 0 at discussion', test_discussion.discussion_id)
            print('At post: ', post.post_id)

        test_data_alignment.append([
            test_discussion.discussion_id,
            post.post_id,
            token_repetition
        ])

    for author in test_unique_authors:
        if author != post.username:
            vocab[author] += response_preprocessed


print(test_data_alignment)

post_ids_for_example = [35, 36, 37]
for post_id in  post_ids_for_example:
    post = test_discussion.posts[post_id]
    preprocessed_message = preprocessed_posts[str(test_discussion.discussion_id) + '-' + str(post.post_id)]
    print(f'postId: {post.post_id} \t author: {post.username} \t message: {post.message} \t preprocessed: {preprocessed_message}')
    print(f'Overlapping lemma\'s: {tokens_repeated_per_post[post_id]}')

    # vocab = {a: vocab_a + response_preprocessed for a, vocab_a in vocab if a != post.username}
    # vocab[post.username] = vocab_not_author

#%% Load ks per bin
ks_per_bin = [4, 6, 5, 5, 4, 5, 5, 4]


#%% Print graphs of best models
for i in range(0, 8):
    k = ks_per_bin[i]
    bin_lengths = lengths_for_bins[i]
    disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

    # Plot and save the best clustering:
    best_model = best_models[i]
    best_model_y_ra = best_model['y_ra']
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
    best_n = best_model['try_counter']

    fig_best, axs_best = plt.subplots(math.ceil(len(best_model_unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_lengths[-1] / 50)), 4 * k))
    fig_best.tight_layout()
    fig_best.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    for i_class, u_class in enumerate(best_model_unique_classes_ra):
        discussions_with_class = best_model_predicted_classes_ra.loc[best_model_predicted_classes_ra['class'] == u_class]
        discussion_ids_with_class = discussions_with_class.index
        discussions_df_with_class = discussions_in_bin_length.loc[
            discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
        discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

        ax_x = math.floor(i_class/2)
        ax_y = i_class % 2
        ax = axs_best[ax_x, ax_y]
        ax.set_ylim((0, 1))
        ax.set_xlim((0, bin_lengths[-1]))
        for d_idx in discussion_ids_with_class:
            discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
            ax.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
        trend = discussions_pivoted_df_with_class.mean()
        mean_per_post = trend.reset_index()
        ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
        # ax.set_xticks([])

    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2/2), 0].set_ylabel(
        'Time-based word repetition per class with rolling averages')
    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2), 0].set_xlabel('Posts in time')
    axs_best[math.floor((len(best_model_unique_classes_ra)-1)/2), 1].set_xlabel('Posts in time')

    fig_best.suptitle(f'Alignment over time for bin {i+1} (lengths {bin_lengths[0]}-{bin_lengths[-1]})')
    fig_best.savefig(f'Results/Clustering/Best_Result/test/line_alignment_bin_{i+1}_attempt_{best_n+1}')
    fig_best.show()


#%% Get "actual" classes of figures
ks_per_bin = [4, 6, 5, 5, 4, 5, 5, 4]
for i in range(0, 8):
    k = ks_per_bin[i]
    bin_lengths = lengths_for_bins[i]
    disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

    # Plot and save the best clustering:
    best_model = best_models[i]
    best_model_y_ra = best_model['y_ra']
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
    print(f'{i+1}: {best_model_unique_classes_ra}')


#%% Get discussion df from class of bin of best model.
to_inspect_bin = 3 #[0, 7]
to_inspect_class = 3

specific_best_model = best_models[to_inspect_bin]

# Get bin df
bin_lengths = lengths_for_bins[to_inspect_bin]
disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]

# Get class df
specific_best_model_predicted_classes = specific_best_model['predicted_classes_ra']
discussions_with_class = specific_best_model_predicted_classes.loc[specific_best_model_predicted_classes['class'] == to_inspect_class]
discussion_ids_with_class = discussions_with_class.index
discussions_df_with_class = discussions_in_bin_length.loc[
    discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]


#%% Get alignment data necessary for interplay analysis
# Loop through bins
alignment_discussion_data = []
class_counter = 0
for bin_id in range(0, 8):
    # Get best clustering model of bin
    best_model = best_models[bin_id]                                                    # Get best model
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']                # Get predicted classes
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()    # Get unique predicted classes

    # Get data per class
    for i_class, u_class in enumerate(best_model_unique_classes_ra):
        discussions_with_class = best_model_predicted_classes_ra.loc[best_model_predicted_classes_ra['class'] == u_class]   # Find discussions with class
        discussion_ids_with_class = discussions_with_class.index                        # Get discussion ids for discussions in class

        # Get data per discussion
        for d_idx in discussion_ids_with_class:
            average_alignment_value_df = average_df.loc[average_df['discussion_id'] == d_idx]['average_alignment']
            average_alignment_value = average_alignment_value_df.iloc[0]
            alignment_class = class_counter
            length_discussion_value_df = length_df.loc[length_df['discussion_id'] == d_idx]['no_posts']
            length_discussion_value = length_discussion_value_df.iloc[0]
            alignment_discussion_data.append([
                d_idx,
                bin_id,
                length_discussion_value,
                average_alignment_value,
                class_counter,
                u_class
            ])

        # Update class counter
        class_counter += 1

# Create df of discussion alignment data
print('[TASK] storing alignment data')
alignment_discussion_data_df = pd.DataFrame(alignment_discussion_data,
                                  columns=['discussion_id', 'bin_id', 'discussion_length', 'average_alignment', 'alignment_class_overall', 'alignment_class_in_bin'])
alignment_discussion_data_df.to_csv(__csv_alignment_class_data__)
print('[INFO] task completed')



#%% Extract deltas of classes
ranges = {}
delta_data = []
for i in range(0, 8):
    ranges[i] = {}
    bin_ids_bin = bin_ids[i]
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(bin_ids_bin)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')
    k = ks_per_bin[i]
    bin_length = lengths_for_bins[i]

    # Plot the best clustering:
    best_model = best_models[i]
    best_model_y_ra = best_model['y_ra']
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
    best_n = best_model['try_counter']

    fig_best, axs_best = plt.subplots(math.ceil(len(best_model_unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_length[1] / 50)), 4 * k))
    fig_best.tight_layout()
    fig_best.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    for i_class in range(0, k):
        discussions_with_class = best_model_predicted_classes_ra.loc[
            best_model_predicted_classes_ra['class'] == i_class]
        discussion_ids_with_class = discussions_with_class.index
        discussions_df_with_class = discussions_in_bin_length.loc[
            discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
        discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

        ax_x = math.floor(i_class / 2)
        ax_y = i_class % 2
        ax = axs_best[ax_x, ax_y]
        ax.set_ylim((0, 1))
        ax.set_xlim((0, bin_length[-1]))
        ax.title.set_text(f'Class {i_class + 1}')
        trend = discussions_pivoted_df_with_class.mean()
        mean_per_post = trend.reset_index()
        ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
        # ax.set_xticks([])

        # remove nan rows
        mean_per_post.dropna(subset=[0], inplace=True)
        # compute deltas
        ranges[i][i_class] = [mean_per_post.iloc[0][0], mean_per_post.iloc[-1][0]]
        delta = mean_per_post.iloc[-1][0] - mean_per_post.iloc[0][0]
        delta_data.append([
            i,
            i_class,
            delta
        ])

    axs_best[math.floor((len(best_model_unique_classes_ra) - 1) / 2 / 2), 0].set_ylabel(
        'Mean time-based word overlap per class')
    axs_best[math.floor((len(best_model_unique_classes_ra) - 1) / 2), 0].set_xlabel('Posts in time')
    axs_best[math.floor((len(best_model_unique_classes_ra) - 1) / 2), 1].set_xlabel('Posts in time')

    fig_best.suptitle(f'Alignment trends over time for bin {i + 1} (lengths {bin_length[0]}-{bin_length[-1]})')
    fig_best.savefig(f'Results/Clustering/trends/best_alignment_bin_{i + 1}.png')
    fig_best.show()

# Create df of delta alignment data
print('[TASK] storing alignment delta data')
alignment_class_delta_df = pd.DataFrame(delta_data,
                                  columns=['bin_id', 'class_id', 'delta'])
alignment_class_delta_df.to_csv(__csv_alignment_class_delta_data__)
print('[INFO] task completed')


#%% Print graphs of specific class per bin
i_bin = 3
u_class = 4

k = ks_per_bin[i_bin]
bin_lengths = lengths_for_bins[i_bin]
disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(disc_ids_with_length)]
pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

# Plot and save the best clustering:
best_model = best_models[i_bin]
best_model_y_ra = best_model['y_ra']
best_model_predicted_classes_ra = best_model['predicted_classes_ra']
best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
best_n = best_model['try_counter']

fig_best, ax_best = plt.subplots(figsize=(4 + (0.5 * (bin_lengths[-1] / 50)), 6))
fig_best.tight_layout()
fig_best.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

discussions_with_class = best_model_predicted_classes_ra.loc[best_model_predicted_classes_ra['class'] == u_class]
discussion_ids_with_class = discussions_with_class.index
discussions_df_with_class = discussions_in_bin_length.loc[
    discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

ax_best.set_ylim((0, 1))
ax_best.set_xlim((0, bin_lengths[-1]))
for d_idx in discussion_ids_with_class:
    discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
    ax_best.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
trend = discussions_pivoted_df_with_class.mean()
mean_per_post = trend.reset_index()
ax_best.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')

ax_best.set_ylabel('Time-based word overlap from rolling averages')
ax_best.set_xlabel('Posts in time')

fig_best.suptitle(f'Alignment over time for bin {i_bin+1}, class {u_class+1}')
fig_best.savefig(f'Results/Clustering/examples/alignment_bin_{i_bin+1}_class_{u_class+1}')
fig_best.show()
