from transformers import BertTokenizer, BertModel
from scipy import spatial
import pandas as pd
import matplotlib as plt
import numpy as np


"""
Get BERT embedding for a message
"""
def preprocess_message_semantic(message):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    print(tokenizer.model_max_length)
    model = BertModel.from_pretrained("bert-base-cased")
    encoded_input = tokenizer(message, return_tensors='pt', truncation=True, max_length=512) # truncates to 512 words as that is the max length accepted by the model
    return model(**encoded_input)


"""
Preprocesses each message for this analysis
"""
def get_preprocessed_messages_for_semantic(discussions):
    print('[TASK] preprocessing messages')
    preprocessed_posts = {}
    # get all the preprocessed posts
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            preprocessed = preprocess_message_semantic(post.message)
            preprocessed_posts[str(i) + '-' + str(j)] = preprocessed
    print('[INFO] task completed')
    return preprocessed_posts


"""
Computes the actual alignment for each message and each of it's parent messages
"""
def compute_semantic_alignment(discussions, preprocessed_messages, path):
    print('[TASK] computing semantical alignment for all messages and all of their parents')
    data = []
    for i in discussions.keys():
        discussion = discussions[i]
        for j in discussion.posts.keys():
            post = discussion.posts[j]
            response_preprocessed_index = str(discussion.discussion_id) + '-' + str(post.post_id)
            response_preprocessed = preprocessed_messages[response_preprocessed_index] # is vector
            for k in range(0, len(post.thread)):
                initial_post_id = post.thread[k]
                initial_preprocessed_index = str(discussion.discussion_id) + '-' + str(initial_post_id)
                initial_preprocessed = preprocessed_messages[initial_preprocessed_index] # is vector
                alignment = 1 - spatial.distance.cosine(initial_preprocessed, response_preprocessed)
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
    df = pd.DataFrame(data, columns=['discussion_id', 'initial_message_id', 'response_message_id', 'distance', 'semantical_alignment'])
    df.to_csv(path)
    print('[INFO] task completed')

    return discussions


"""
Generates one histogram for all discussions individually, turns per alignment bin
"""
def get_overall_histogram_semantic_alignment(df, path):
    discussion_ids = list(df['discussion_id'].unique())

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 300000) #300000 for linear
    # ax.set_yscale('log')

    ax.set_title('Semantical alignment for all discussions')
    ax.set_xlabel('Alignment as cosine distance')
    ax.set_ylabel('# of posts')

    # alignment is in range [-1, 1], normalize to [0, 1]
    df['semantical_alignment'] = (df['semantical_alignment'] + 1) / 2

    for d_idx in discussion_ids:
        discussion_df = df[df['discussion_id'] == d_idx]
        alignment = discussion_df['semantical_alignment']
        ax.hist(alignment, bins=np.arange(0, 1, 0.025), alpha=0.1, label=str(d_idx)) #color='#d74a94'  histtype='step'
        print(d_idx)
    # sns.displot(df, x='lexical_word_alignment', hue='discussion_id', kde=True)

    fig.savefig(path)
    fig.show()
