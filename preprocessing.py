from Helpers import read_csv, print_t, print_i, store_data
from Models.Discussion import Discussion
from Models.Post import Post
from datetime import datetime
import re


__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'
__pickle_path_preprocessed_time_based_linear__ = './PickleData/preprocessed_time_based_linear'


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

#%% Remove empty discussions
# print_t('Removing discussions with empty messages')
# empty_discussion_indices = []
# for i in discussions.keys():
#     discussion = discussions[i]
#     for j in discussion.posts.keys():
#         post = discussion.posts[j]
#         text_ = post.message
#         if not text_ or not isinstance(text_, str):
#             # message has no text, remove discussion.
#             empty_discussion_indices.append(discussion.discussion_id)
#
# print_i('Removing ' + str(len(empty_discussion_indices)) + 'discussions...')
# stripped_discussions = {key: value for key, value in zip(discussions.keys(), discussions.values()) if value.discussion_id not in empty_discussion_indices }
#
# print_i('Removed empty discussions, ' + str(len(stripped_discussions.keys())) + ' discussions left')


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


#%% store preprocessed data
store_data(discussions, __pickle_path_preprocessed_time_based_linear__)

