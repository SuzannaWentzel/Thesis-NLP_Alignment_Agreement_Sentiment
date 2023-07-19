#%% imports
from Helpers import print_t, print_i, store_data
import pandas as pd
import matplotlib.pyplot as plt
# from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize
# from nltk.corpus import wordnet, stopwords
# import nltk
# from nltk.stem import WordNetLemmatizer
# import string
# import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tslearn.clustering import TimeSeriesKMeans
import math


# __preprocessing_adapted_LLA_avg_linear__ = 'AlignmentData/preprocessing_LLA_alignment_linear.csv'
__pickle_path_preprocessed__ = './PickleData/preprocessed_time_based_linear'
__pickle_path_df_sentiment_preprocessed__ = './PickleData/preprocessed_df_sentiment'
__pickle_path_preprocessed_sentiment__ = './PickleData/preprocessed_sentiment'
__csv_sentiment_data__ = './AlignmentData/sentiment.csv'
__pickle_path_best_sentiment_clustering_data__ = './PickleData/best_sentiment_clustering_data'
__pickle_path_bin_ids__ = './PickleData/bin_ids'


#%% Load preprocessed data (see preprocessing.py)
# Reset discussions (just necessary for python console)
discussions = {}
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_preprocessed__))
store_file = open(__pickle_path_preprocessed__, 'rb')
discussions = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Preprocess messages
def preprocess_message_sentiment(paragraph):
    """
    Preprocesses text posts before applying VADER, splitting up posts in sentences.
    :param paragraph: message to preprocess
    :return: preprocessed message: list of sentences
    """
    # Tokenize post
    sentences = sent_tokenize(paragraph)
    return sentences


#%% Preprocess all messages for sentiment analysis
print_t('preprocessing messages')
preprocessed_posts = {}
# get all the preprocessed posts
data = []
for i in discussions.keys():
    print(f'Preprocessing for sentiment: {i} out of {len(discussions.keys())}')
    discussion = discussions[i]
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        preprocessed = preprocess_message_sentiment(post.message)
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
df.to_pickle(__pickle_path_df_sentiment_preprocessed__)
print_i('stored preprocessing data')


#%% Store preprocessed messages for lexical alignment analysis
store_data(preprocessed_posts, __pickle_path_preprocessed_sentiment__)


#%% Load prerprocessed messages
discussions = {}
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_preprocessed_sentiment__))
store_file = open(__pickle_path_preprocessed_sentiment__, 'rb')
preprocessed_posts = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Sentiment function
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(sentences):
    """
    Returns the compound VADER sentiment
    :param sentences: list of sentences (strings)
    :return: compound VADER sentiment
    """
    sentiments = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        sentiments.append(vs['compound'])
    mean_sentiment = np.mean(sentiments)
    return mean_sentiment


#%% Compute sentiment for all posts
print_t('computing sentiment for all messages')
data_sentiment = []
for i in discussions.keys():
    print('computing sentiment', i)
    discussion = discussions[i]
    discussion_df = df.loc[df['discussion_id'] == i]

    for j in discussion.posts.keys():
        post = discussion.posts[j]
        response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
        response_preprocessed = preprocessed_posts[response_preprocessed_index]

        sentiment = get_sentiment(response_preprocessed)

        data_sentiment.append([
            discussion.discussion_id,
            post.post_id,
            sentiment
        ])


#%% Storing sentiment data
print('[TASK] storing sentiment data')
sentiment_df = pd.DataFrame(data_sentiment, columns=['discussion_id', 'post_id', 'compound_sentiment'])
sentiment_df.to_csv(__csv_sentiment_data__)
print('[INFO] task completed')


#%% Load sentiment data
sentiment_df = pd.read_csv(__csv_sentiment_data__)


#%% Plot data
discussion_length = []
unique_disc_idxs = sentiment_df['discussion_id'].unique()

for d_idx in unique_disc_idxs:
    discussion = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    sentiment_vals = discussion['compound_sentiment']
    p_idxs = range(1, len(sentiment_vals) + 1)

    plt.plot(p_idxs, sentiment_vals, linewidth=0.5)
    # plt.scatter(p_idxs, alignment_vals, color='#d74a94', s=1)

plt.xlabel('time (in posts)')
plt.xlim((0, 1103))
plt.ylim((-1, 1))
plt.ylabel('Sentiment')
plt.suptitle('Sentiment over time')
plt.savefig('Results/Sentiment/Time/sentiment_time_all.png')
plt.show()


#%% Plot sentiment for all messages distribution
print('mean: \t', sentiment_df['compound_sentiment'])
print('percentiles: \t', sentiment_df['compound_sentiment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2, ax3) = plt.subplots(3)

fig.suptitle('Sentiment in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(-1, 1)
ax2.set_xlim(-1, 1)
ax3.set_xlim(-1, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# posts')
ax3.set_ylim(0, 6300)
ax3.set_xlabel('Sentiment')

ax1.hist(sentiment_df['compound_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(sentiment_df['compound_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax3.hist(sentiment_df['compound_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.savefig('Results/Sentiment/histo_sentiment_messages.png')
fig.show()


#%% Get average sentiment
averages = []
for d_idx in unique_disc_idxs:
    print('averaging sentiment', d_idx)
    discussion_df = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    discussion_sentiment_avg = discussion_df['compound_sentiment'].mean()
    averages.append([
        d_idx,
        discussion_sentiment_avg
    ])

average_df = pd.DataFrame(averages, columns=['discussion_id', 'average_sentiment'])


#%% Plot sentiment distribution
print('mean: \t', average_df['average_sentiment'].mean())
print('percentiles: \t', average_df['average_sentiment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Average sentiment in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# discussions')
ax2.set_xlabel('Mean sentiment')

ax1.hist(average_df['average_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(average_df['average_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.savefig('Results/Sentiment/histo_sentiment_discussions.png')
fig.show()


#%% Inspect sentiment bins
discussion_spikes_low = average_df[average_df['average_sentiment'] < -0.375]
print('amount of discussions in spikes in beginning: ', len(discussion_spikes_low))
print('alignment spikes: ', discussion_spikes_low.to_string())

discussion_1 = average_df[average_df['average_sentiment'] < average_df['average_sentiment'].quantile(.01)]
sample_discussion_1 = discussion_1.sample(n=10, random_state=1)
print('Random sample of discussions with alignment in first percentile: ', sample_discussion_1.to_string())

discussion_50 = average_df[(average_df['average_sentiment'] < average_df['average_sentiment'].quantile(.51)) & (average_df['average_sentiment'] > average_df['average_sentiment'].quantile(.49))]
sample_discussion_50 = discussion_50.sample(n=10, random_state=1)
print('Random sample of discussions with alignment at median:', sample_discussion_50.to_string())

discussion_995 = average_df[average_df['average_sentiment'] > average_df['average_sentiment'].quantile(.995)]
sample_discussion_995 = discussion_995.sample(n=10, random_state=1)
print('Random sample of discussions with alignment in last 0.05th percentile: ', sample_discussion_995.to_string())

discussion_spikes_high = average_df[average_df['average_sentiment'] > 0.3]
print('amount of discussions in spikes on end: ', len(discussion_spikes_high))
print('alignment spikes: ', discussion_spikes_high.to_string())


#%% Get median sentiment
medians = []
for d_idx in unique_disc_idxs:
    print('getting median sentiment', d_idx)
    discussion_df = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    discussion_sentiment_avg = discussion_df['compound_sentiment'].median()
    medians.append([
        d_idx,
        discussion_sentiment_avg
    ])

median_df = pd.DataFrame(medians, columns=['discussion_id', 'median_sentiment'])


#%% Plot sentiment distribution of median
print('mean: \t', median_df['median_sentiment'].mean())
print('percentiles: \t', median_df['median_sentiment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Median sentiment in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# discussions')
ax2.set_xlabel('Median sentiment')

ax1.hist(median_df['median_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(median_df['median_sentiment'], bins=np.arange(-1, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.show()


#%% Get variance of sentiment
variances = []
for d_idx in unique_disc_idxs:
    print('getting variances', d_idx)
    discussion_df = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    discussion_sentiment_variance = discussion_df['compound_sentiment'].var()
    variances.append([
        d_idx,
        discussion_sentiment_variance
    ])

variance_df = pd.DataFrame(variances, columns=['discussion_id', 'variance_sentiment'])


#%% Plot sentiment distribution variances
print('mean: \t', variance_df['variance_sentiment'].mean())
print('percentiles: \t', variance_df['variance_sentiment'].describe(percentiles=[0, .01, .05, .1, .9, .95, .99, .995, 1]))

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Variance of sentiment in discussions')
fig.subplots_adjust(hspace=0.5)

ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_yscale('log')
ax2.set_ylabel('# discussions')
ax2.set_xlabel('Variance in sentiment')

ax1.hist(variance_df['variance_sentiment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'
ax2.hist(variance_df['variance_sentiment'], bins=np.arange(0, 1, 0.01),
         color='#d74a94')  # color='#d74a94'  histtype='step'

fig.show()


#%% Plot message sentiment for 5 discussions
discussion_length = []
unique_disc_idxs = sentiment_df['discussion_id'].unique()

for d_idx in [1, 2, 3, 4, 7]:
    discussion = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    sentiment_vals = discussion['compound_sentiment']
    p_idxs = range(1, len(sentiment_vals) + 1)

    plt.plot(p_idxs, sentiment_vals, linewidth=0.5)
    # plt.scatter(p_idxs, alignment_vals, color='#d74a94', s=1)

plt.xlabel('time (in posts)')
plt.xlim((0, 100))
plt.ylim((-1, 1))
plt.ylabel('Sentiment')
plt.suptitle('Sentiment over time')
plt.savefig('Results/Sentiment/Time/sentiment_time_all.png')
plt.show()


#%% Obtain timeseries clustering
# # add correct post index
# df_times = alignment_df.copy()
# for d_idx in df_times['discussion_id'].unique():
#     discussion = df_times.loc[df_times['discussion_id'] == d_idx]
#     df_times.loc[df_times['discussion_id'] == d_idx, 'time_post_id'] = range(0, len(discussion))
#
# df_times = df_times.drop(columns='post_id')
# pivoted = df_times.pivot(index='discussion_id', columns='time_post_id')
#
#
# model = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10)
# y = model.fit_predict(pivoted.values)
# x = df_times['time_post_id'].unique()
#
#
# #%% Plot time series
# predicted_classes = pd.DataFrame(data=y, index=pivoted.index, columns=['class'])
# unique_classes = predicted_classes['class'].unique()
#
# fig, axs = plt.subplots(len(unique_classes))
#
# for u_class in unique_classes:
#     ax = axs[u_class]
#     discussions_with_class = predicted_classes.loc[predicted_classes['class'] == u_class]
#     discussion_ids_with_class = discussions_with_class.index
#     discussions_df_with_class = df_times.loc[df_times['discussion_id'].isin(discussion_ids_with_class)]
#     for d_idx in discussion_ids_with_class:
#         discussion = df_times.loc[df_times['discussion_id'] == d_idx]
#         ax.plot(discussion['time_post_id'], discussion['lexical_word_alignment'])
#
# axs[int(np.median(unique_classes))].set_ylabel('Lexical word alignment')
# axs[len(unique_classes)-1].set_xlabel('Posts in time')
# fig.suptitle('Alignment over time per found class')
# fig.show()
#
#
# #%% Plot time series with trends
# predicted_classes = pd.DataFrame(data=y, index=pivoted.index, columns=['class'])
# unique_classes = predicted_classes['class'].unique()
#
# fig, axs = plt.subplots(len(unique_classes))
# fig.subplots_adjust(hspace=0.3)
# fig.set_size_inches(8.5, 10)
#
#
# for u_class in unique_classes:
#     ax = axs[u_class]
#     discussions_with_class = predicted_classes.loc[predicted_classes['class'] == u_class]
#     discussion_ids_with_class = discussions_with_class.index
#     discussions_df_with_class = df_times.loc[df_times['discussion_id'].isin(discussion_ids_with_class)]
#     discussions_pivoted_df_with_class = pivoted.loc[discussion_ids_with_class]
#     for d_idx in discussion_ids_with_class:
#         discussion = df_times.loc[df_times['discussion_id'] == d_idx]
#         ax.plot(discussion['time_post_id'], discussion['lexical_word_alignment'])
#     trend = discussions_pivoted_df_with_class.mean()
#     mean_per_post = trend.reset_index()
#     ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
#     # ax.set_xticks([])
#
# axs[int(np.median(unique_classes))].set_ylabel('Time-based word repetition per class')
# axs[len(unique_classes)-1].set_xlabel('Posts in time')
#
# fig.suptitle('Alignment over time per found class')
# fig.show()
#
#
#%% Cumsum rolling average
def rolling_average(a, n=3):
    ret = np.cumsum(a, dtype=float) # computes cumulative sum for
    ret[n:] = ret[n:] - ret[:-n] # subtract cumsum before window from cumsum at end of window to get the sum inside the window (bc of the shift)
    res = ret[n - 1:] / n # divide by windowsize, throw away values outside of possible windows
    return res


#%% Plot trends
unique_disc_idxs = sentiment_df['discussion_id'].unique()

window_size = 5
data_rolling_average = []
for d_idx in unique_disc_idxs:
    discussion = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
    alignment_vals = discussion['compound_sentiment']
    p_idxs = range(1 + (window_size//2), len(alignment_vals) + 1 - (window_size//2))
    rolling_averages = rolling_average(alignment_vals.values, n=window_size)
    plt.plot(p_idxs, rolling_averages, linewidth=0.5)
    data_rolling_average_disc = [[d_idx, p_idxs[i], rolling_averages[i]] for i in range(0, len(p_idxs))]
    data_rolling_average += data_rolling_average_disc

plt.xlabel('time (in posts)')
plt.xlim((0, 1103))
plt.ylim((-1, 1))
plt.ylabel('Rolling average of sentiment')
plt.suptitle('Rolling average of sentiment over time')
plt.savefig('Results/Sentiment/Time/sentiment_RA_time_all.png')
plt.show()

rolling_average_df = pd.DataFrame(data_rolling_average, columns=['discussion_id', 'time_post_id', 'rolling_average'])

#%% for 5 dicscussions:
# discussion_length = []
# unique_disc_idxs = sentiment_df['discussion_id'].unique()
#
# window_size = 5
# data_rolling_average = []
# for d_idx in [1, 2, 3, 4, 7]:
#     discussion = sentiment_df.loc[sentiment_df['discussion_id'] == d_idx]
#     alignment_vals = discussion['compound_sentiment']
#     p_idxs = range(1 + (window_size//2), len(alignment_vals) + 1 - (window_size//2))
#     rolling_averages = rolling_average(alignment_vals.values, n=window_size)
#     plt.plot(p_idxs, rolling_averages, linewidth=0.5)
#     data_rolling_average_disc = [[d_idx, p_idxs[i], rolling_averages[i]] for i in range(0, len(p_idxs))]
#     data_rolling_average += data_rolling_average_disc
#
# plt.xlabel('time (in posts)')
# plt.xlim((0, 100))
# plt.ylim((-1, 1))
# plt.ylabel('Rolling average of sentiment')
# plt.suptitle('Rolling average of sentiment over time')
# plt.savefig('Results/Sentiment/Time/sentiment_RA_time_all.png')
# plt.show()
#
# rolling_average_df = pd.DataFrame(data_rolling_average, columns=['discussion_id', 'time_post_id', 'rolling_average'])


#%% Apply clustering to rolling average
# pivoted_rolling_average = rolling_average_df.pivot(index='discussion_id', columns='time_post_id')
# model_rolling_avg = TimeSeriesKMeans(n_clusters=15, metric="dtw", max_iter=10)
# y_ra = model_rolling_avg.fit_predict(pivoted_rolling_average.values)
# x_ra = rolling_average_df['time_post_id'].unique()
#
# #%% Plot clustered classes with rolling averages
# predicted_classes_ra = pd.DataFrame(data=y_ra, index=pivoted_rolling_average.index, columns=['class'])
# unique_classes_ra = predicted_classes_ra['class'].unique()
#
# fig, axs = plt.subplots(len(unique_classes_ra))
# fig.subplots_adjust(hspace=1)
# fig.set_size_inches(8.5, 50)
#
# for u_class in unique_classes_ra:
#     ax = axs[u_class]
#     ax.set_ylim((0, 1))
#     ax.set_xlim((0, 1300))
#     discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
#     discussion_ids_with_class = discussions_with_class.index
#     discussions_df_with_class = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(discussion_ids_with_class)]
#     discussions_pivoted_df_with_class = pivoted_rolling_average.loc[discussion_ids_with_class]
#     for d_idx in discussion_ids_with_class:
#         discussion = rolling_average_df.loc[rolling_average_df['discussion_id'] == d_idx]
#         ax.plot(discussion['time_post_id'], discussion['rolling_average'])
#     trend = discussions_pivoted_df_with_class.mean()
#     mean_per_post = trend.reset_index()
#     ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
#     # ax.set_xticks([])
#
# axs[int(np.median(unique_classes_ra))].set_ylabel('Time-based word repetition per class with rolling averages')
# axs[len(unique_classes_ra)-1].set_xlabel('Posts in time')
#
# fig.suptitle('Alignment over time per found class')
# fig.show()
#
#


#%% Load bins that were used before
print_t('Loading preprocessed data from pickle path ' + str(__pickle_path_bin_ids__))
store_file = open(__pickle_path_bin_ids__, 'rb')
bins_ids = pickle.load(store_file)
store_file.close()
print_i('Loaded data from pickle')


#%% Load bin lengths
bin_lengths = [(7, 22), (23, 33), (34, 50), (51, 86), (87, 136), (137, 186), (187, 236), (237, 286)] # TODO: not hardcode this list

#%% Inertias for different k's for each bin
# Go through length bins
for i, bin_ids in enumerate(bins_ids):
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(bin_ids)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')
    max_no_posts = bin_lengths[i][1]

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
        fig, axs = plt.subplots(math.ceil(len(unique_classes_ra) / 2), 2, figsize=(8+(1 * (max_no_posts / 50)), 8*(math.ceil(n/2))))
        if n == 1:
            fig, axs = plt.subplots(1, figsize=(8+(1 * (max_no_posts / 50)), 8*(math.ceil(n/2))))

        fig.tight_layout()
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        for i_class, u_class in enumerate(unique_classes_ra):
            discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
            discussion_ids_with_class = discussions_with_class.index
            discussions_df_with_class = discussions_in_bin_length.loc[
                discussions_in_bin_length['discussion_id'].isin(discussion_ids_with_class)]
            discussions_pivoted_df_with_class = pivoted_discussions_in_bin_length.loc[discussion_ids_with_class]

            if n == 1:
                ax = axs
            elif n == 2:
                ax = axs[i_class]
            else:
                ax_x = math.floor(i_class/2)
                ax_y = i_class % 2
                ax = axs[ax_x, ax_y]
            ax.set_ylim((-1, 1))
            ax.set_xlim((0, max_no_posts))
            for d_idx in discussion_ids_with_class:
                discussion = discussions_in_bin_length.loc[discussions_in_bin_length['discussion_id'] == d_idx]
                ax.plot(discussion['time_post_id'], discussion['rolling_average'], linewidth=0.7)
            trend = discussions_pivoted_df_with_class.mean()
            mean_per_post = trend.reset_index()
            ax.plot(mean_per_post['time_post_id'], mean_per_post[0], linestyle='dashed', color='black')
            # ax.set_xticks([])

        if n == 1:
            axs.set_ylabel('Sentiment score per class with rolling averages')
            axs.set_xlabel('Posts in time')
        elif n == 2:
            axs[0].set_ylabel(
                'Sentiment score per class with rolling averages')
            axs[0].set_xlabel('Posts in time')
            axs[1].set_xlabel('Posts in time')
        else:
            axs[math.floor((len(unique_classes_ra)-1)/2/2), 0].set_ylabel(
                'Sentiment score per class with rolling averages')
            axs[math.floor((len(unique_classes_ra)-1)/2), 0].set_xlabel('Posts in time')
            axs[math.floor((len(unique_classes_ra)-1)/2), 1].set_xlabel('Posts in time')

        fig.suptitle(f'Sentiment over time for bin {i + 1}, k={n}')
        fig.savefig(f'Results/Sentiment/Clustering/5_bins_and_last_splitted/sentiment_bin_{i+1}_k_{n}')
        fig.show()

    fig_elbow, ax_elbow = plt.subplots(figsize=(4, 4))
    fig_elbow.tight_layout()
    fig_elbow.subplots_adjust(top=0.9, left=0.2, right=0.95, bottom=0.15)
    ax_elbow.plot(ks, inertias, color='#d74a94')
    ax_elbow.set_xlabel('number of clusters (k)')
    ax_elbow.set_ylabel('inertia')
    fig_elbow.suptitle(f'Clustering for bin {i+1} (length {bin_lengths[i][0]} - {bin_lengths[i][1]})')
    fig_elbow.savefig(f'Results/Sentiment/Clustering/5_bins_and_last_splitted/line_sos_bin_{i+1}')
    fig_elbow.show()


#%% Cluster per bin
ks_per_bin = [6, 5, 7, 6, 8, 6, 7, 6]
tries = 5
best_models = []
all_models = []

# Go through length bins
for i, bin_ids in enumerate(bins_ids):
    discussions_in_bin_length = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(bin_ids)]
    pivoted_discussions_in_bin_length = discussions_in_bin_length.pivot(index='discussion_id', columns='time_post_id')

    # for each bin, try out different ks and compute mean sum of squares
    inertias = []
    k = ks_per_bin[i]
    best_model = {}
    bin_length = bin_lengths[i]

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
        fig, axs = plt.subplots(math.ceil(len(unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_length[1] / 50)), 4 * k))
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
            ax.set_ylim((-1, 1))
            ax.set_xlim((0, bin_length[1]))
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
        fig.savefig(f'Results/Sentiment/Clustering/attempt2/line_alignment_bin_{i+1}_attempt_{n+1}')
        fig.show()

    # store the best model
    best_models.append(best_model)

    # Plot and save the best clustering:
    best_model_y_ra = best_model['y_ra']
    best_model_predicted_classes_ra = best_model['predicted_classes_ra']
    best_model_unique_classes_ra = best_model_predicted_classes_ra['class'].unique()
    best_n = best_model['try_counter']

    fig_best, axs_best = plt.subplots(math.ceil(len(best_model_unique_classes_ra)/2), 2, figsize=(8 + (1 * (bin_length[1] / 50)), 4 * k))
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
        ax.set_ylim((-1, 1))
        ax.set_xlim((0, bin_length[1]))
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

    fig_best.suptitle(f'Alignment over time for bin {i+1} (lengths {bin_length[0]}-{bin_length[1]})')
    fig_best.savefig(f'Results/Sentiment/Clustering/attempt2/best_line_alignment_bin_{i+1}_attempt_{best_n+1}')
    fig_best.show()
#
#
# #%% Get amount of discussions per bin
# for i, bin_lengths in enumerate(lengths_for_bins):
#     bin_counter += 1
#     disc_ids_with_length = length_df.loc[length_df['no_posts'].isin(bin_lengths)]['discussion_id'].unique()
#     print_i(f'Bin {i + 1}: \t {len(disc_ids_with_length)} discussions')
#
#
#%% Store best found models
store_data(best_models, __pickle_path_best_sentiment_clustering_data__)
