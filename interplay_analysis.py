#%% Imports
import pandas as pd
from Helpers import print_i
import matplotlib.pyplot as plt
from scipy.stats.contingency import association
import seaborn as sns


__csv_alignment_class_data__ = './AlignmentData/alignment_classes.csv'
__csv_sentiment_class_data__ = './SentimentData/sentiment_classes.csv'
__csv_topic_data__ = './Data/discussion_topic.csv'
__csv_sentiment_class_delta_data__ = './SentimentData/sentiment_classes_deltas.csv'
__csv_alignment_class_delta_data__ = './AlignmentData/alignment_classes_deltas.csv'



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
min_sentiment = sentiment_class_df['min_sentiment']
max_sentiment = sentiment_class_df['max_sentiment']
sentiment_class_overall = sentiment_class_df['sentiment_class_overall']
sentiment_class_in_bin = sentiment_class_df['sentiment_class_in_bin']
topic = topic_df['topic']
interplay_df = pd.concat([interplay_df, average_sentiment], axis=1)
interplay_df = pd.concat([interplay_df, min_sentiment], axis=1)
interplay_df = pd.concat([interplay_df, max_sentiment], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_overall], axis=1)
interplay_df = pd.concat([interplay_df, sentiment_class_in_bin], axis=1)
interplay_df = pd.concat([interplay_df, topic], axis=1)

interplay_df['discussion_id'] = interplay_df.index
interplay_df = interplay_df.drop('Unnamed: 0', axis=1)

#%% Compute Pearson correlation
print_i(f'Pearson Correlation [average alignment - discussion length]: {interplay_df["average_alignment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [average sentiment - discussion length]: {interplay_df["average_sentiment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [min sentiment - discussion length]: {interplay_df["min_sentiment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [max sentiment - discussion length]: {interplay_df["max_sentiment"].corr(interplay_df["discussion_length"])}')
print_i(f'Pearson Correlation [average alignment - average sentiment]: {interplay_df["average_alignment"].corr(interplay_df["average_sentiment"])}')
print_i(f'Pearson Correlation [average alignment - min sentiment]: {interplay_df["average_alignment"].corr(interplay_df["min_sentiment"])}')
print_i(f'Pearson Correlation [average alignment - max sentiment]: {interplay_df["average_alignment"].corr(interplay_df["max_sentiment"])}')

#%% Plot correlation
# ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='discussion_length', color='#d74a94', s=1)
# ax1.set_xlabel('Average time-based overlap')
# ax1.set_ylabel('Discussion length (# posts)')
# plt.savefig('./Results/Interplay/cor_alignment_length.png')

p = sns.jointplot(data=interplay_df, x="average_alignment", y="discussion_length", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Average time-based overlap", ylabel="Discussion length (# posts)")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_alignment_length_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='average_sentiment', y='discussion_length', color='#d74a94', s=1)
# ax1.set_xlabel('Average sentiment score')
# ax1.set_ylabel('Discussion length (# posts)')
# plt.savefig('./Results/Interplay/cor_sentiment_length.png')

p = sns.jointplot(data=interplay_df, x="average_sentiment", y="discussion_length", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Average sentiment score", ylabel="Discussion length (# posts)")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_sentiment_length_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='min_sentiment', y='discussion_length', color='#d74a94', s=1)
# ax1.set_xlabel('Min sentiment score')
# ax1.set_ylabel('Discussion length (# posts)')
# plt.savefig('./Results/Interplay/cor_min_sentiment_length.png')

p = sns.jointplot(data=interplay_df, x="min_sentiment", y="discussion_length", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Min sentiment score", ylabel="Discussion length (# posts)")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_min_sentiment_length_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='max_sentiment', y='discussion_length', color='#d74a94', s=1)
# ax1.set_xlabel('Max sentiment score')
# ax1.set_ylabel('Discussion length (# posts)')
# plt.savefig('./Results/Interplay/cor_max_sentiment_length.png')

p = sns.jointplot(data=interplay_df, x="max_sentiment", y="discussion_length", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Max sentiment score", ylabel="Discussion length (# posts)")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_max_sentiment_length_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='average_sentiment', color='#d74a94', s=1)
# ax1.set_xlabel('Average time-based overlap')
# ax1.set_ylabel('Average sentiment score')
# plt.savefig('./Results/Interplay/cor_alignment_sentiment.png')

p = sns.jointplot(data=interplay_df, x="average_alignment", y="average_sentiment", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Average time-based overlap", ylabel="Average sentiment score")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_alignment_sentiment_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='min_sentiment', color='#d74a94', s=1)
# ax1.set_xlabel('Average time-based overlap')
# ax1.set_ylabel('Min sentiment score')
# plt.savefig('./Results/Interplay/cor_alignment_min_sentiment.png')

p = sns.jointplot(data=interplay_df, x="average_alignment", y="min_sentiment", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Average time-based overlap", ylabel="Min sentiment score")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_alignment_min_sentiment_heat.png')
plt.close()

# ax1 = interplay_df.plot(kind='scatter', x='average_alignment', y='max_sentiment', color='#d74a94', s=1)
# ax1.set_xlabel('Average time-based overlap')
# ax1.set_ylabel('Max sentiment score')
# plt.savefig('./Results/Interplay/cor_alignment_max_sentiment.png')

p = sns.jointplot(data=interplay_df, x="average_alignment", y="max_sentiment", height=5, ratio=2, color="#d74a94", kind="hist")
p.set_axis_labels(xlabel="Average time-based overlap", ylabel="Max sentiment score")
p.ax_marg_x.remove()
p.ax_marg_y.remove()
plt.tight_layout()
plt.savefig('./Results/Interplay/cor_alignment_max_sentiment_heat.png')
plt.close()



#%% Make contingency tables per bin
cross_tabs_alignment_sentiment = {}
cross_tabs_alignment_topic = {}
cross_tabs_sentiment_topic = {}
for bin_id in range(0, 8):
    interplay_bin_df = interplay_df.loc[interplay_df['bin_id'] == bin_id]
    cross_tabs_alignment_sentiment[bin_id] = pd.crosstab(interplay_bin_df['alignment_class_in_bin'], interplay_bin_df['sentiment_class_in_bin'])
    cross_tabs_alignment_topic[bin_id] = pd.crosstab(interplay_bin_df['alignment_class_in_bin'], interplay_bin_df['topic'])
    cross_tabs_sentiment_topic[bin_id] = pd.crosstab(interplay_bin_df['sentiment_class_in_bin'], interplay_bin_df['topic'])

#%%
cross_tabs_alignment_discussions = interplay_df[['alignment_class_in_bin', 'topic', 'discussion_id']]
cross_tabs_alignment_discussions = cross_tabs_alignment_discussions[cross_tabs_alignment_discussions['topic'].notna()]
cross_tabs_alignment_discussions = cross_tabs_alignment_discussions.set_index(['alignment_class_in_bin', 'topic'])

cross_tabs_sentiment_discussions = interplay_df[['sentiment_class_in_bin', 'topic', 'discussion_id']]
cross_tabs_sentiment_discussions = cross_tabs_sentiment_discussions[cross_tabs_sentiment_discussions['topic'].notna()]
cross_tabs_sentiment_discussions = cross_tabs_sentiment_discussions.set_index(['sentiment_class_in_bin', 'topic'])

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


#%% Remove columns contingency tables
cross_tab_alignment_topic_reduced_data = {}
cross_tab_sentiment_topic_reduced_data = {}

for bin_id in range(0, 8):
    # Get which column sums are smaller than amount of rows
    cross_tab_alignment_topic_bin_df = cross_tabs_alignment_topic[bin_id]
    no_rows_align = len(cross_tab_alignment_topic_bin_df.index)
    to_remove_cols_align = []
    for name, col in cross_tab_alignment_topic_bin_df.items():
        if col.sum() < no_rows_align:
            to_remove_cols_align.append(name)

    # Remove columns that are too sparse
    reduced_alignment_topic_cross_tab_df = cross_tab_alignment_topic_bin_df.drop(to_remove_cols_align, axis=1, inplace=False)

    # Find rows that now sum to 0
    to_remove_rows_align = []
    for index, row in reduced_alignment_topic_cross_tab_df.iterrows():
        if row.sum() == 0:
            to_remove_rows_align.append(index)

    # Remove rows that sum to 0
    double_reduced_alignment_topic_cross_tab_df = reduced_alignment_topic_cross_tab_df.drop(index=to_remove_rows_align, inplace=False)
    cross_tab_alignment_topic_reduced_data[bin_id] = double_reduced_alignment_topic_cross_tab_df

    # Obtain which columns are too sparse
    cross_tab_sentiment_topic_bin_df = cross_tabs_sentiment_topic[bin_id]
    no_rows_sent = len(cross_tab_sentiment_topic_bin_df.index)
    to_remove_cols_sent = []
    for name, col in cross_tab_sentiment_topic_bin_df.items():
        if col.sum() < no_rows_sent:
            to_remove_cols_sent.append(name)

    # Remove columns that are too sparse
    reduced_sentiment_topic_cross_tab_df = cross_tab_sentiment_topic_bin_df.drop(to_remove_cols_sent, axis=1, inplace=False)

    # Find rows that sum to 0 now
    to_remove_rows_sent = []
    for index, row in reduced_sentiment_topic_cross_tab_df.iterrows():
        if row.sum() == 0:
            to_remove_rows_sent.append(index)

    # Remove rows that sum to 0
    double_reduced_sentiment_topic_cross_tab_df = reduced_sentiment_topic_cross_tab_df.drop(index=to_remove_rows_sent, inplace=False)
    cross_tab_sentiment_topic_reduced_data[bin_id] = double_reduced_sentiment_topic_cross_tab_df


#%% Compute cramer's V again
# Compute and print Cramer's V for alignment and topic per bin
for bin_id in range(0, 8):
    reduced_cross_tab_alignment_topic_bin_df = cross_tab_alignment_topic_reduced_data[bin_id]
    cramers_v_alignment_topic = association(reduced_cross_tab_alignment_topic_bin_df, method="cramer")
    print_i(f'Cramer\'s V for alignment class and topic for bin {bin_id + 1}: {cramers_v_alignment_topic}')

# Compute and print Cramer's V for sentiment and topic per bin
for bin_id in range(0, 8):
    reduced_cross_tab_sentiment_topic_bin_df = cross_tab_sentiment_topic_reduced_data[bin_id]
    cramers_v_sentiment_topic = association(reduced_cross_tab_sentiment_topic_bin_df, method="cramer")
    print_i(f'Cramer\'s V for sentiment class and topic for bin {bin_id + 1}: {cramers_v_sentiment_topic}')


#%% Load delta data
sentiment_class_delta_df = pd.read_csv(__csv_sentiment_class_delta_data__) # contains length data
alignment_class_delta_df = pd.read_csv(__csv_alignment_class_delta_data__)


#%% Order contingency tables
cross_tabs_alignment_topic_ordered = {}
cross_tabs_sentiment_topic_ordered = {}
for bin_id in range(0, 8):
    # Get delta data and index correclty
    sentiment_delta_bin_df = sentiment_class_delta_df.loc[sentiment_class_delta_df['bin_id'] == bin_id]
    sentiment_delta_bin_df = sentiment_delta_bin_df.set_index('class_id')

    # Get cross table and add delta column
    reduced_cross_tab_sentiment_topic_bin_df = cross_tab_sentiment_topic_reduced_data[bin_id]
    sentiment_delta_column = sentiment_delta_bin_df['delta']
    sentiment_combined_cross_tab = pd.concat([reduced_cross_tab_sentiment_topic_bin_df, sentiment_delta_column], axis=1)
    sentiment_sent_sorted_cross_tab = sentiment_combined_cross_tab.sort_values(by='delta')
    cross_tabs_sentiment_topic_ordered[bin_id] = sentiment_sent_sorted_cross_tab

    # Get alignment things
    # Get delta data and index correclty
    alignment_delta_bin_df = alignment_class_delta_df.loc[alignment_class_delta_df['bin_id'] == bin_id]
    alignment_delta_bin_df = alignment_delta_bin_df.set_index('class_id')

    # Get cross table and add delta column
    reduced_cross_tab_alignment_topic_bin_df = cross_tab_alignment_topic_reduced_data[bin_id]
    alignment_delta_column = alignment_delta_bin_df['delta']
    alignment_combined_cross_tab = pd.concat([reduced_cross_tab_alignment_topic_bin_df, alignment_delta_column], axis=1)
    alignment_sorted_cross_tab = alignment_combined_cross_tab.sort_values(by='delta')
    cross_tabs_alignment_topic_ordered[bin_id] = alignment_sorted_cross_tab

#%% Order contingency table sentiment vs alignment
cross_tabs_alignment_sentiment_ordered = {}
for bin_id in range(0, 8):
    # Get alignment things
    # Get delta data and index correclty
    alignment_delta_bin_df = alignment_class_delta_df.loc[alignment_class_delta_df['bin_id'] == bin_id]
    alignment_delta_bin_df = alignment_delta_bin_df.set_index('class_id')

    # Get cross table and add delta column
    cross_tab_alignment_sentiment_bin_df = cross_tabs_alignment_sentiment[bin_id]
    alignment_delta_column = alignment_delta_bin_df['delta']
    alignment_combined_cross_tab = pd.concat([cross_tab_alignment_sentiment_bin_df, alignment_delta_column], axis=1)
    alignment_sorted_cross_tab = alignment_combined_cross_tab.sort_values(by='delta', axis=0)

    # Get delta data and index correclty
    sentiment_delta_bin_df = sentiment_class_delta_df.loc[sentiment_class_delta_df['bin_id'] == bin_id]
    sentiment_delta_bin_df = sentiment_delta_bin_df.set_index('class_id')

    # Get cross table and add delta column
    sentiment_delta_column = sentiment_delta_bin_df['delta']
    sentiment_combined_cross_tab = pd.concat([alignment_sorted_cross_tab, pd.DataFrame(sentiment_delta_column).T])
    sentiment_sorted_cross_tab = sentiment_combined_cross_tab.sort_values(by='delta', axis=1)

    cross_tabs_alignment_sentiment_ordered[bin_id] = sentiment_sorted_cross_tab


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



