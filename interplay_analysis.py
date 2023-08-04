#%% Imports
import pandas as pd
from Helpers import print_i
import matplotlib.pyplot as plt



__csv_alignment_class_data__ = './AlignmentData/alignment_classes.csv'
__csv_sentiment_class_data__ = './SentimentData/sentiment_classes.csv'


#%% Load data
alignment_class_df = pd.read_csv(__csv_alignment_class_data__) # contains length data
sentiment_class_df = pd.read_csv(__csv_sentiment_class_data__)


#%% Combine data
alignment_class_df = alignment_class_df.set_index('discussion_id')
sentiment_class_df = sentiment_class_df.set_index('discussion_id')

interplay_df = alignment_class_df.copy()
average_sentiment = sentiment_class_df['average_sentiment']
sentiment_class_overall = sentiment_class_df['sentiment_class_overall']
sentiment_class_in_bin = sentiment_class_df['sentiment_class_in_bin']
interplay_df = pd.concat([interplay_df, average_sentiment], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_overall], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_in_bin], axis=1)

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
for bin_id in range(0, 8):
    interplay_bin_df = interplay_df.loc[interplay_df['bin_id'] == bin_id]
    cross_tabs_alignment_sentiment[bin_id] = pd.crosstab(interplay_bin_df['alignment_class_in_bin'], interplay_bin_df['sentiment_class_in_bin'])


#%% Compute cramers V for categorical variables
# https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables
