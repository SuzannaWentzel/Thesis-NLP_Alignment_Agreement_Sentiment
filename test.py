from Helpers import read_csv, print_t, print_i
from Models.Discussion import Discussion
from Models.Post import Post
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import nltk
from nltk.stem import WordNetLemmatizer


__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'


#%% Read data
data = read_csv(__datapath__)


# Run preprocessing

#%% Converts discussion dataframe to list of objects
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


#%% Get linear threads
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


#%% Remove empty messages
for i in discussions.keys():
    discussion = discussions[i]
    to_remove_posts = []
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        message = post.message
        if not message or not isinstance(message, str):
            print_i(f'Found post to remove in discussion_id {discussion.discussion_id} and post_id {post.post_id}')
            to_remove_posts.append(post.post_id)

    # Remove the posts
    for j in to_remove_posts:
        del discussion.posts[j]

    # update threads to not include the to remove messages
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        new_thread = [post_id for post_id in post.thread if post_id not in to_remove_posts]
        post.set_thread(new_thread)


#%% Merge consecutive messages
print_t('merging consecutive or empty messages')
# Find message where previous message is of the same author
to_remove_posts_total = []
to_remove_posts_from_discussions = set()
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
                to_remove_posts_total.append(j)
                to_remove_posts_from_discussions.add(i)

    # Replace all thread histories where that id occured with the previous message id to keep the threads intact
    for k in to_remove_posts:
        del discussion.posts[k]

    for j in discussion.posts.keys():
        post = discussion.posts[j]
        new_threads = [indx for indx in post.thread if indx not in to_remove_posts]
        post.set_thread(new_threads)

print_i(f'Found {len(to_remove_posts_total)} posts to merge from {len(to_remove_posts_from_discussions)} discussions')
print_i('merged consecutive messages')


#%% Replace URLs
print_t('Replacing URLs with [URL] tags')
for i in discussions.keys():
    discussion = discussions[i]
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        message = re.sub('http[s]?://\S+', '[URL]', post.message)
        message = re.sub('www.\S+', '[URL]', message)
        post.update_message(message)
print_i('replaced URLs with [URL] tags')


#%% Remove discussions that have now less than two authors with at least 4 posts
print_t('Removing discussions that have less than two authors with at least 4 posts')
discussions_to_remove = []
for i in discussions.keys():
    author_list = {}
    discussion = discussions[i]
    for j in discussion.posts.keys():
        post = discussion.posts[j]
        if not post.username in author_list.keys():
            author_list[post.username] = 1
        else:
            author_list[post.username] += 1

    authors_enough = 0
    for a in author_list.keys():
        if author_list[a] >= 4:
            authors_enough += 1

    if authors_enough < 2:
        discussions_to_remove.append(i)

print_i(f'Found {len(discussions_to_remove)} discussions to remove: {discussions_to_remove}')

print_t('Removing too short discussions')
for i in discussions_to_remove:
    del discussions[i]

print_i('Removed too short discussions')
print_i(f'Amount of discussions left: {len(discussions.keys())}')


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


def preprocess_message_lexical_word(post_to_preprocess):
    """
    Preprocesses text posts before applying LILLA, by tokenizing, lowercasing and removing separate punctuation tokens.
    :param post_to_preprocess: message to preprocess
    :return: preprocessed message: list of lemmas
    """
    # Tokenize post
    tokens = word_tokenize(post_to_preprocess)
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

print_t('storing preprocessing data to df')
df = pd.DataFrame(data, columns=['discussion_id', 'post_id', 'author_id', 'text', 'preprocessed_text'])
print_i('stored preprocessing data')


#%% Removing empty posts
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

print('[TASK] storing alignment data to df')
alignment_df = pd.DataFrame(data_alignment, columns=['discussion_id', 'post_id', 'lexical_word_alignment'])
print('[INFO] task completed')


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


#%% Get average alignment
averages = []
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
