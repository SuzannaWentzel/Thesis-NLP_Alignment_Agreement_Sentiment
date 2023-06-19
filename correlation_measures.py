

__avg_alignment_linear_LLA__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_LLA.csv'
__avg_alignment_linear_Jaccard__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_Jaccard.csv'
__avg_alignment_linear_SCP__ = './Results/DataStats/TempStorage/df_avg_alignment_linear_SCP.csv'

from Helpers import read_csv, print_i


def get_correlations():
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


get_correlations()
