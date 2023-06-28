from Helpers import read_csv, print_i
import pandas as pd


__avg_alignment_linear_LLA__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_LLA.csv'
__avg_alignment_linear_Jaccard__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_Jaccard.csv'
__avg_alignment_linear_SCP__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_SCP.csv'
__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_four_posts.csv'




def get_correlations_alignment_measures():
    avg_alignment_LLA = read_csv(__avg_alignment_linear_LLA__)
    avg_alignment_LLA.set_index('discussion_id')
    avg_alignment_Jaccard = read_csv(__avg_alignment_linear_Jaccard__)
    avg_alignment_Jaccard.set_index('discussion_id')
    avg_alignment_SCP = read_csv(__avg_alignment_linear_SCP__)
    avg_alignment_SCP.set_index('discussion_id')

    # unique_discussions = avg_alignment_SCP['discussion_id'].unique()
    # avg_alignment_LLA_filtered = avg_alignment_LLA[avg_alignment_LLA['discussion_id'].isin(unique_discussions)]
    # avg_alignment_Jaccard_filtered = avg_alignment_Jaccard[avg_alignment_Jaccard['discussion_id'].isin(unique_discussions)]

    # avg_alignment_LLA_filtered.set_index('discussion_id')
    # avg_alignment_Jaccard_filtered.set_index('discussion_id')

    print_i('Pearson Correlation LLA/Jaccard: ' + str(avg_alignment_LLA['average_alignment'].corr(avg_alignment_Jaccard['average_alignment'])))
    print_i('Pearson Correlation Jaccard/SCP: ' + str(avg_alignment_Jaccard['average_alignment'].corr(avg_alignment_SCP['average_alignment'])))
    print_i('Pearson Correlation LLA/SCP: ' + str(avg_alignment_LLA['average_alignment'].corr(avg_alignment_SCP['average_alignment'])))


# get_correlations_alignment_measures()


def get_correlation_length_adapted_lla():






    avg_alignment_LLA = read_csv(__avg_alignment_linear_LLA__)
    avg_alignment_LLA.set_index('discussion_id')
    discussion_length = []
    discussions = read_csv(__datapath__)
    unique_disc_idxs = discussions['discussion_id'].unique()

    for d_idx in unique_disc_idxs:
        print('at ', d_idx)
        discussion = discussions.loc[discussions['discussion_id'] == d_idx]
        discussion_length.append([
            d_idx, len(discussion)
        ])
    length_df = pd.DataFrame(discussion_length, columns=['discussion_id', 'no_posts'])
    length_df.set_index('discussion_id')

    print_i('Pearson Correlation LLA/discussion length: ' + str(avg_alignment_LLA['average_alignment'].corr(length_df['no_posts'])))

get_correlation_length_adapted_lla()