#%% imports
from Helpers import read_csv, print_t, print_i
from datetime import datetime
from Models.Discussion import Discussion
from Models.Post import Post
import re
import copy
from linguistic_alignment_analysis.compute_lexical_word_alignment import preprocess_message_lexical_word, \
    adapted_LLA
import pandas as pd
from main import store_data
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from numpy.polynomial import Chebyshev
from sklearn.metrics import davies_bouldin_score

__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'
__preprocessing_adapted_LLA_avg_linear__ = 'AlignmentData/preprocessing_LLA_alignment_linear.csv'
__pickle_path_preprocessed_linear__ = './PickleData/preprocessed_linear'
__pickle_path_df_lexical_word_preprocessed_linear__ = './PickleData/preprocessed_df_lexical_word_linear'
__csv_alignment_data__ = './AlignmentData/lexical_alignment.csv'



#%% Read data
data = read_csv(__datapath__)


#%% Run preprocessing

# Converts discussion dataframe to list of objects
print_t('Converting dataframe into discussions')
discussions = {}
discussion_indices = data['discussion_id'].unique()
for i in discussion_indices:
    discussion_df = data.loc[data['discussion_id'] == i]
    posts = {}
    for index, row in discussion_df.iterrows():
        date = datetime.strptime(str(row['creation_date']), "%Y-%m-%d %H:%M:%S")
        post = Post(row['discussion_id'], row['post_id'], row['text'], row['parent_post_id'], row['author_id'], date)
        posts[row['post_id']] = post
    discussion = Discussion(i, posts)
    discussions[i] = discussion
print_i('Converted df into discussions')

# Remove empty discussions
print_t('Removing discussions with empty messages')
empty_discussion_indices = []
for i in discussions.keys():
    discussion = discussions[i]
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        text_ = post.message
        if not text_ or not isinstance(text_, str):
            # message has no text, remove discussion.
            empty_discussion_indices.append(discussion.discussion_id)

print_i('Removing ' + str(len(empty_discussion_indices)) + 'discussions...')
stripped_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in empty_discussion_indices }

print_i('Removed empty discussions, ' + str(len(stripped_discussions.keys())) + ' discussions left')

# Replace URLs
print_t('Replacing URLs with [URL] tags')
for i in stripped_discussions.keys():
    discussion = stripped_discussions[i]
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        message = re.sub('http[s]?://\S+', '[URL]', post.message)
        message = re.sub('www.\S+', '[URL]', message)
        post.update_message(message)
print_i('replaced URLs with [URL] tags')

replaced_urls = copy.deepcopy(stripped_discussions)

# Get linear threads
print_t('getting linear discussions')
# posts are with post_id already ordered by date in discussions
# create thread with all previous posts
for i in replaced_urls.keys():
    discussion = replaced_urls[i]
    post_list = list(discussion.posts.keys())
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        history = post_list[:post_list.index(j)]
        post.set_thread(history)

print_i('got linear discussions')

# Merge consecutive messages
print_t('merging consecutive messages')
# Find message where previous message is of the same author
for i in replaced_urls.keys():
    discussion = replaced_urls[i]
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


#%% store preprocessed data
store_data(replaced_urls, __pickle_path_preprocessed_linear__)


#%% Preprocess messages for lexical alignment
print_t('preprocessing messages')
preprocessed_posts = {}
# get all the preprocessed posts
data = []
for i in replaced_urls.keys():
    print(f'Preprocessing for lexical word: {i} out of {len(replaced_urls.keys())}')
    discussion = replaced_urls[i]
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


#%% Compute alignment
print_t('computing lexical word alignment for all messages and all of their parents')
data_alignment = []

for i in replaced_urls.keys():
    print('computing alignment', i)
    discussion = replaced_urls[i]
    discussion_df = df.loc[df['discussion_id'] == i]
    # print('1', discussion_df)

    if len(discussion_df) == 50:
        unique_authors = df['author_id'].unique()
        vocab = {a: [] for a in unique_authors}
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_posts[response_preprocessed_index]
            vocab_not_author = vocab[post.username]
            if len(vocab_not_author) != 0:
                tokens_repeated = [word for word in response_preprocessed if word in vocab_not_author]
                token_repetition = len(tokens_repeated) / len(response_preprocessed)
                data_alignment.append([
                    discussion.discussion_id,
                    post.post_id,
                    token_repetition
                ])
            else:
                print('Found 0 at discussion', discussion.discussion_id)
                print('At post: ', post.post_id)

            for author in unique_authors:
                if author != post.username:
                    vocab[author] += response_preprocessed

            # vocab = {a: vocab_a + response_preprocessed for a, vocab_a in vocab if a != post.username}
            # vocab[post.username] = vocab_not_author

print('[TASK] storing alignment data')
alignment_df = pd.DataFrame(data_alignment, columns=['discussion_id', 'post_id', 'lexical_word_alignment'])
alignment_df.to_csv(__csv_alignment_data__)
print('[INFO] task completed')




#%% Plot data
discussion_length = []
unique_disc_idxs = alignment_df['discussion_id'].unique()

for d_idx in unique_disc_idxs:
    print('at ', d_idx)
    discussion = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    discussion_length.append([
        d_idx, len(discussion) + 1
    ])
length_df = pd.DataFrame(discussion_length, columns=['discussion_id', 'no_posts'])
disc_length_50 = length_df.loc[length_df['no_posts'] == 50]
print(disc_length_50)

fig = plt.figure()

discussion_50_alignment = alignment_df.loc[alignment_df['discussion_id'].isin(disc_length_50['discussion_id'])]
unique_disc_idxs = discussion_50_alignment['discussion_id'].unique()
for d_idx in unique_disc_idxs:
    discussion = discussion_50_alignment.loc[discussion_50_alignment['discussion_id'] == d_idx]
    alignment_vals = discussion['lexical_word_alignment']
    p_idxs = range(1, len(alignment_vals) + 1)

    plt.plot(p_idxs, alignment_vals)
    # plt.scatter(p_idxs, alignment_vals, color='#d74a94', s=1)

plt.xlabel('time (in posts)')
plt.xlim((0, 50))
plt.ylabel('Time-based word repetition')
plt.show()
# plt.savefig(storage_path)


#%% Obtain timeseries clustering
# add correct post index
df_times = discussion_50_alignment.copy()
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

for d_idx in unique_disc_idxs:
    print('at ', d_idx)
    discussion = alignment_df.loc[alignment_df['discussion_id'] == d_idx]
    discussion_length.append([
        d_idx, len(discussion) + 1
    ])
length_df = pd.DataFrame(discussion_length, columns=['discussion_id', 'no_posts'])
disc_length_50 = length_df.loc[length_df['no_posts'] == 50]
print(disc_length_50)

window_size = 5

data_rolling_average = []
discussion_50_alignment = alignment_df.loc[alignment_df['discussion_id'].isin(disc_length_50['discussion_id'])]
unique_disc_idxs = discussion_50_alignment['discussion_id'].unique()
for d_idx in unique_disc_idxs:
    discussion = discussion_50_alignment.loc[discussion_50_alignment['discussion_id'] == d_idx]
    alignment_vals = discussion['lexical_word_alignment']
    p_idxs = range(1 + (window_size//2), len(alignment_vals) + 1 - (window_size//2))
    rolling_averages = rolling_average(alignment_vals.values, n=window_size)
    plt.plot(p_idxs, rolling_averages)
    data_rolling_average_disc = [[d_idx, p_idxs[i], rolling_averages[i]] for i in range(0, len(p_idxs))]
    data_rolling_average += data_rolling_average_disc

plt.xlabel('time (in posts)')
plt.xlim((0, 50))
plt.ylim((0, 1))
plt.ylabel('Rolling average of time-based word repetition')
plt.show()
# plt.savefig(storage_path)

rolling_average_df = pd.DataFrame(data_rolling_average, columns=['discussion_id', 'time_post_id', 'rolling_average'])

#%% Apply clustering to rolling average
pivoted_rolling_average = rolling_average_df.pivot(index='discussion_id', columns='time_post_id')
model_rolling_avg = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10)
y_ra = model_rolling_avg.fit_predict(pivoted_rolling_average.values)
x_ra = rolling_average_df['time_post_id'].unique()

#%% Plot clustered classes with rolling averages
predicted_classes_ra = pd.DataFrame(data=y_ra, index=pivoted_rolling_average.index, columns=['class'])
unique_classes_ra = predicted_classes_ra['class'].unique()

fig, axs = plt.subplots(len(unique_classes_ra))
fig.subplots_adjust(hspace=0.3)
fig.set_size_inches(8.5, 10)

for u_class in unique_classes_ra:
    ax = axs[u_class]
    ax.set_ylim((0, 1))
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


#%% Try out metrics
# Doesnt work bc of the time:
# davies_bouldin_score = davies_bouldin_score(rolling_average_df, y_ra)
# print(davies_bouldin_score)

# find inertia, returned by kmeans
inertia = model_rolling_avg.inertia_
print(inertia)

# Find number of clusters by elbow method.
metrics = []
no_clusters = []
inertia = []
for n in range(3, 10):
    model_rolling_avg = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=5)
    y_ra = model_rolling_avg.fit_predict(pivoted_rolling_average.values)
    x_ra = rolling_average_df['time_post_id'].unique()
    metrics.append([
        n,
        model_rolling_avg,
        model_rolling_avg.inertia_
    ])
    no_clusters.append(n)
    inertia.append(model_rolling_avg.inertia_)

plt.plot(no_clusters, inertia)
plt.show()


#%% Inter class variance & cardinality

variances = []
cardinality = []
means_sum_squares = []
for u_class in unique_classes_ra:
    discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
    discussion_ids_with_class = discussions_with_class.index
    discussions_df_with_class = rolling_average_df.loc[rolling_average_df['discussion_id'].isin(discussion_ids_with_class)]
    discussions_pivoted_df_with_class = pivoted_rolling_average.loc[discussion_ids_with_class]

    variances_timeseries = discussions_pivoted_df_with_class.var(axis=0)
    mean_class_variance = variances_timeseries.mean()
    variances.append(mean_class_variance)
    cardinality.append(discussions_pivoted_df_with_class.index.size)

    mean_timeseries = discussions_pivoted_df_with_class.mean(axis=0)
    diffs_timeseries = discussions_pivoted_df_with_class.sub(mean_timeseries)
    squared_timeseries = diffs_timeseries.pow(2)
    sum_time = squared_timeseries.sum(axis=0)
    mean_sum_squares = sum_time.mean()
    means_sum_squares.append(mean_sum_squares)

plt.figure()
plt.scatter(cardinality, variances)
plt.show()

total_means_sos = sum(means_sum_squares)
max_variance = max(variances)
min_cardinality = min(cardinality)
print(f'Total mean of sum of squares: {total_means_sos}')
print(f'Max variance: {max_variance}')
print(f'Min cardinality: {min_cardinality}')

#%% SOS for different k's

mean_soss = []
ks = [k for k in range(1, 10)]
for n in ks:
    model_rolling_avg = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10)
    y_ra = model_rolling_avg.fit_predict(pivoted_rolling_average.values)
    x_ra = rolling_average_df['time_post_id'].unique()
    metrics.append([
        n,
        model_rolling_avg,
        model_rolling_avg.inertia_
    ])
    no_clusters.append(n)
    inertia.append(model_rolling_avg.inertia_)

    variances = []
    cardinality = []
    means_sum_squares = []
    for u_class in y_ra:
        discussions_with_class = predicted_classes_ra.loc[predicted_classes_ra['class'] == u_class]
        discussion_ids_with_class = discussions_with_class.index
        discussions_df_with_class = rolling_average_df.loc[
        rolling_average_df['discussion_id'].isin(discussion_ids_with_class)]
        discussions_pivoted_df_with_class = pivoted_rolling_average.loc[discussion_ids_with_class]

        variances_timeseries = discussions_pivoted_df_with_class.var(axis=0)
        mean_class_variance = variances_timeseries.mean()
        variances.append(mean_class_variance)
        cardinality.append(discussions_pivoted_df_with_class.index.size)

        mean_timeseries = discussions_pivoted_df_with_class.mean(axis=0)
        diffs_timeseries = discussions_pivoted_df_with_class.sub(mean_timeseries)
        squared_timeseries = diffs_timeseries.pow(2)
        sum_time = squared_timeseries.sum(axis=0)
        mean_sum_squares = sum_time.mean()
        means_sum_squares.append(mean_sum_squares)

    plt.figure()
    plt.scatter(cardinality, variances)
    plt.show()

    total_means_sos = sum(means_sum_squares)
    max_variance = max(variances)
    min_cardinality = min(cardinality)
    # mean_soss.append(total_means_sos)
    mean_soss.append(sum(variances))

plt.plot(ks, mean_soss)
plt.show()
