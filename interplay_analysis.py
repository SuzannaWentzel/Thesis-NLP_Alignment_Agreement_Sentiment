#%% Imports
import pandas as pd
from Helpers import print_i
import matplotlib.pyplot as plt
from scipy.stats.contingency import association
import numpy as np


__csv_alignment_class_data__ = './AlignmentData/alignment_classes.csv'
__csv_sentiment_class_data__ = './SentimentData/sentiment_classes.csv'
__csv_topic_data__ = './Data/discussion_topic.csv'


#%% Load data
alignment_class_df = pd.read_csv(__csv_alignment_class_data__) # contains length data
sentiment_class_df = pd.read_csv(__csv_sentiment_class_data__)
topic_df = pd.read_csv(__csv_topic_data__)


#%% Combine data

# Filter topic data on discussions only in other dfs
unique_disc_ids = alignment_class_df['discussion_id'].unique()
topic_df = topic_df.loc[topic_df['discussion_id'].isin(unique_disc_ids)]

# Set indices
alignment_class_df = alignment_class_df.set_index('discussion_id')
sentiment_class_df = sentiment_class_df.set_index('discussion_id')
topic_df = topic_df.set_index('discussion_id')


interplay_df = alignment_class_df.copy()
average_sentiment = sentiment_class_df['average_sentiment']
sentiment_class_overall = sentiment_class_df['sentiment_class_overall']
sentiment_class_in_bin = sentiment_class_df['sentiment_class_in_bin']
topic = topic_df['topic']
interplay_df = pd.concat([interplay_df, average_sentiment], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_overall], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_in_bin], axis=1)
interplay_df = pd.concat([interplay_df, topic], axis=1)

interplay_df['discussion_id'] = interplay_df.index
interplay_df = interplay_df.drop('Unnamed: 0', axis=1)

#%% Compute Pearson correlation
print_i(f'Pearson Correlation [average alignment - discussion length]: {interplay_df["average_alignment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [average sentiment - discussion length]: {interplay_df["average_sentiment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [average alignment - average sentiment]: {interplay_df["average_alignment"].corr(interplay_df["average_sentiment"])}')

#%% Plot correlation
ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='discussion_length', color='#d74a94', s=1)
ax1.set_xlabel('Average time-based overlap')
ax1.set_ylabel('Discussion length (# posts)')
plt.savefig('./Results/Interplay/cor_alignment_length.png')

ax1 = interplay_df.plot(kind='scatter', x='average_sentiment', y='discussion_length', color='#d74a94', s=1)
ax1.set_xlabel('Average sentiment score')
ax1.set_ylabel('Discussion length (# posts)')
plt.savefig('./Results/Interplay/cor_sentiment_length.png')

ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='average_sentiment', color='#d74a94', s=1)
ax1.set_xlabel('Average time-based overlap')
ax1.set_ylabel('Average sentiment score')
plt.savefig('./Results/Interplay/cor_alignment_sentiment.png')


#%% Make contingency tables per bin
cross_tabs_alignment_sentiment = {}
cross_tabs_alignment_topic = {}
cross_tabs_sentiment_topic = {}
for bin_id in range(0, 8):
    interplay_bin_df = interplay_df.loc[interplay_df['bin_id'] == bin_id]
    cross_tabs_alignment_sentiment[bin_id] = pd.crosstab(interplay_bin_df['alignment_class_in_bin'], interplay_bin_df['sentiment_class_in_bin'])
    cross_tabs_alignment_topic[bin_id] = pd.crosstab(interplay_bin_df['alignment_class_in_bin'], interplay_bin_df['topic'])
    cross_tabs_sentiment_topic[bin_id] = pd.crosstab(interplay_bin_df['sentiment_class_in_bin'], interplay_bin_df['topic'])

#%% Compute cramers V for categorical variables
# https://towardsdatascience.com/contingency-tables-chi-squared-and-cramers-v-ada4f93ec3fd
# https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html

# Compute and print Cramer's V for alignment and sentiment per bin
for bin_id in range(0, 8):
    cross_tab_alignment_sentiment_bin_df = cross_tabs_alignment_sentiment[bin_id]
    cramers_v_alignment_sentiment = association(cross_tab_alignment_sentiment_bin_df, method="cramer")
    print_i(f'Cramer\'s V for alignment and sentiment classes for bin {bin_id + 1}: {cramers_v_alignment_sentiment}')

# Compute and print Cramer's V for alignment and topic per bin
for bin_id in range(0, 8):
    cross_tab_alignment_topic_bin_df = cross_tabs_alignment_topic[bin_id]
    cramers_v_alignment_topic = association(cross_tab_alignment_topic_bin_df, method="cramer")
    print_i(f'Cramer\'s V for alignment class and topic for bin {bin_id + 1}: {cramers_v_alignment_topic}')

# Compute and print Cramer's V for sentiment and topic per bin
for bin_id in range(0, 8):
    cross_tab_sentiment_topic_bin_df = cross_tabs_sentiment_topic[bin_id]
    cramers_v_sentiment_topic = association(cross_tab_sentiment_topic_bin_df, method="cramer")
    print_i(f'Cramer\'s V for sentiment class and topic for bin {bin_id + 1}: {cramers_v_sentiment_topic}')


#%% Compute Association for all discussions (overview)

# Compute contingency tables
cross_tabs_alignment_sentiment_all = pd.crosstab(interplay_df['alignment_class_overall'], interplay_df['sentiment_class_overall'])
cross_tabs_alignment_topic_all = pd.crosstab(interplay_df['alignment_class_overall'], interplay_df['topic'])
cross_tabs_sentiment_topic_all = pd.crosstab(interplay_df['sentiment_class_overall'], interplay_df['topic'])

# Compute Cramer's V
cramers_v_alignment_sentiment_all = association(cross_tabs_alignment_sentiment_all, method="cramer")
cramers_v_alignment_topic_all = association(cross_tabs_alignment_topic_all, method="cramer")
cramers_v_sentiment_topic_all = association(cross_tabs_sentiment_topic_all, method="cramer")

# Print results
print_i(f'Cramer\'s V for alignment and sentiment classes for all discussions: {cramers_v_alignment_sentiment_all}')
print_i(f'Cramer\'s V for alignment class and topic for bin all discussions: {cramers_v_alignment_topic_all}')
print_i(f'Cramer\'s V for sentiment class and topic for bin all discussions: {cramers_v_sentiment_topic_all}')


#%% Figure out how many discussions per bin are annotated with topic

# All instances of discussions without topic
# discs_without_topic = interplay_df.loc[(not interplay_df['topic']) | (not isinstance(interplay_df['topic'], str)]
nan_count = 0
disc_count = 0
for row_i, row in interplay_df.iterrows():
    disc_count += 1
    topic = row['topic']
    if not topic or not isinstance(topic, str):
        nan_count += 1

print_i(f'{nan_count} out of {disc_count} discussions are not annotated with topic')
print_i(f'{disc_count - nan_count} are annotated with topic')

# Count discussions per bin without topic
nan_counts = {}
disc_counts = {}
for bin_id in range(0, 8):
    interplay_bin_df = interplay_df.loc[interplay_df['bin_id'] == bin_id]
    disc_count = 0
    nan_count = 0
    for row_i, row in interplay_bin_df.iterrows():
        disc_count += 1
        topic = row['topic']
        if not topic or not isinstance(topic, str):
            nan_count += 1
    nan_counts[bin_id] = nan_count
    disc_counts[bin_id] = disc_count
    print_i(f'Bin {bin_id + 1}: {nan_count} out of {disc_count} discussions are not annotated with topic')
    print_i(f'Bin {bin_id + 1}: {disc_count - nan_count} are annotated with topic')

