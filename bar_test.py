import matplotlib as plt
import numpy as np
import pandas as pd

from Helpers import read_csv

__datapath__filtered = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'

#%%
discussions = read_csv(__datapath__filtered)

#%%
test = discussions.head(100)
discussion_indices = test['discussion_id'].unique()
idx = discussion_indices[0]
discussion = test.loc[test['discussion_id'] == idx]

#%%
posts_per_author = discussion.groupby('author_id', as_index=False)['post_id'].count().rename(columns={'post_id' : 'post_count'})

#%%

discussion_posts = discussion['post_id'].count()
posts_per_author['aandeel'] = posts_per_author['post_count'] / discussion_posts