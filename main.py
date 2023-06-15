from Helpers import read_csv, print_t, print_i
from linguistic_alignment_analysis.compute_lexical_word_alignment import get_preprocessed_messages_for_lexical_word, \
    compute_lexical_word_alignment, get_overall_histogram_lexical_word_alignment, \
    get_overall_histogram_lexical_word_alignment_stacked
from linguistic_alignment_analysis.compute_semantical_alignment import get_preprocessed_messages_for_semantic, \
    compute_semantic_alignment, get_overall_histogram_semantic_alignment
from linguistic_alignment_analysis.compute_syntactical_alignment import get_syntactical_alignment
from linguistic_alignment_analysis.preprocessing_all import run_preprocessing
import pickle
import os


__threaded_data__ = {}
__linear_data__ = {}

# __datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_two_posts.csv'
# __datapath__ = './Data/discussion_post_text_date_author_parents_unfiltered.csv'
__datapath__ = './Data/dummy_data.csv'

__pickle_path_preprocessed_linear__ = './PickleData/preprocessed_linear'
__pickle_path_preprocessed_tree__ = './PickleData/preprocessed_tree'
__pickle_path_preprocessed_for_lexical_word_thread__ = './PickleData/preprocessed_lexical_word_thread'
__pickle_path_preprocessed_for_lexical_word_linear__ = './PickleData/preprocessed_lexical_word_linear'
__pickle_path_preprocessed_for_semantic_linear__ = 'PickleData/preprocessed_semantic_linear'
__pickle_path_preprocessed_for_semantic_thread__ = 'PickleData/preprocessed_semantic_thread'
__pickle_path_df_lexical_word_preprocessed_thread__ = './PickleData/preprocessed_df_lexical_word_thread' # TODO: change this file type
__pickle_path_df_lexical_word_preprocessed_linear__ = './PickleData/preprocessed_df_lexical_word_linear' # TODO: change this file type
__pickle_path_preprocessed_author_data_linear__ = './PickleData/preprocessed_author_data_linear'
__pickle_path_preprocessed_author_data_thread__ = './PickleData/preprocessed_author_data_thread'
__pickle_path_lexical_word_normed_vals_linear__ = './PickleData/lexical_word_normed_vals_linear'
__pickle_path_lexical_word_normed_vals_thread__ = './PickleData/lexical_word_normed_vals_thread'
__csv_lexical_word_alignment_thread__ = './AlignmentData/lexical_word_alignment.csv'
__csv_lexical_word_alignment_linear__ = './AlignmentData/lexical_word_alignment_linear.csv'
__csv_semantic_alignment_thread__ = './AlignmentData/semantic_alignment_thread.csv'
__csv_semantic_alignment_linear__ = './AlignmentData/semantic_alignment_linear.csv'


def store_data(data, path):
    """
    Stores the preprocessed data as pickle to work faster
    :param data: data to pickle
    :param path: path where to store the data
    """
    print_t('Pickling data to ' + str(path))
    store_file = open(path, 'ab')
    pickle.dump(data, store_file)
    store_file.close()
    print_i('Pickled data')


def load_data(path):
    """
    Loads the preprocessed data from pickle to work faster
    :param path: path where pickle is stored
    :return: data from pickle
    """
    print_t('Loading preprocessed data from pickle path ' + str(path))
    store_file = open(path, 'rb')
    data = pickle.load(store_file)
    store_file.close()
    print_i('Loaded data from pickle')
    return data


def get_preprocessed_data():
    """
    Main function for getting the preprocessed data: runs preprocessing and stores them in global var
    """
    print_t('Preprocessing data')
    global __threaded_data__, __linear_data__
    __threaded_data__, __linear_data__ = run_preprocessing(__datapath__)
    store_data(__threaded_data__, __pickle_path_preprocessed_tree__)
    store_data(__linear_data__, __pickle_path_preprocessed_linear__)
    print_i('Preprocessing completed')


def get_preprocessed_data_from_pickle():
    """
    Main function for getting the preprocessed data from a pickle
    :return: stores the pre-computed preprocessed data in global vars
    """
    print_t('Getting preprocessed data')
    global __threaded_data__, __linear_data__

    __threaded_data__ = load_data(__pickle_path_preprocessed_tree__)
    __linear_data__ = load_data(__pickle_path_preprocessed_linear__)

    print_i(f'Len threaded data: {len(__threaded_data__)}')
    print_i(f'Len linear data: {len(__linear_data__)}')
    print_i('Got preprocessed data')


def get_lexical_word_alignment(lexical_preprocessed=True, alignment_ran=True, get_linear=False, get_thread=False):
    """
    Get lexical word alignment
    :param lexical_preprocessed: boolean if discussions have been preprocessed for this analysis
    :param alignment_ran: boolean if alignment has been run
    :param get_linear: boolean if we want to investigate the linear threads
    :param get_thread: boolean if we want to investigate the tree-structured threads
    :return: nothing: shows histograms and prints percentiles
    """
    print_t('Getting lexical word alignment')
    global __threaded_data__, __linear_data__

    if get_linear:
        print_t('Getting linear lexical word alignment')
        # Get linear lexical word alignment
        if not lexical_preprocessed:
            # preprocess messages for linear
            preprocessed_lexical_word_linear = get_preprocessed_messages_for_lexical_word(__linear_data__, __pickle_path_df_lexical_word_preprocessed_linear__, __pickle_path_preprocessed_author_data_linear__)
            store_data(preprocessed_lexical_word_linear, __pickle_path_preprocessed_for_lexical_word_linear__)
        else:
            # load messages for linear
            preprocessed_lexical_word_linear = load_data(__pickle_path_preprocessed_for_lexical_word_linear__)

        if not alignment_ran:
            # run alignment
            alignment_linear_df = compute_lexical_word_alignment(__linear_data__, preprocessed_lexical_word_linear, __csv_lexical_word_alignment_linear__, __pickle_path_preprocessed_author_data_linear__, __pickle_path_lexical_word_normed_vals_linear__)
        else:
            # load alignment
            alignment_linear_df = read_csv(__csv_lexical_word_alignment_linear__)

        get_overall_histogram_lexical_word_alignment_stacked(alignment_linear_df, './Results/Lexical_word_alignment/all_histo_linear_stacked')
        print_i('task completed: got linear lexical word alignment')

    if get_thread:
        print_t('Getting threaded lexical word alignment')
        # Get thread lexical word alignment
        if not lexical_preprocessed:
            # preprocess messages for thread
            preprocessed_lexical_word_thread = get_preprocessed_messages_for_lexical_word(__threaded_data__, __pickle_path_df_lexical_word_preprocessed_thread__, __pickle_path_preprocessed_author_data_thread__)
            store_data(preprocessed_lexical_word_thread, __pickle_path_preprocessed_for_lexical_word_thread__)
        else:
            # load messages for thread
            preprocessed_lexical_word_thread = load_data(__pickle_path_preprocessed_for_lexical_word_thread__)

        if not alignment_ran:
            # run alignment
            alignment_threaded_df = compute_lexical_word_alignment(__threaded_data__, preprocessed_lexical_word_thread, __csv_lexical_word_alignment_thread__, __pickle_path_preprocessed_author_data_thread__, __pickle_path_lexical_word_normed_vals_thread__)
        else:
            # load alignment
            alignment_threaded_df = read_csv(__csv_lexical_word_alignment_thread__)

        get_overall_histogram_lexical_word_alignment_stacked(alignment_threaded_df, './Results/Lexical_word_alignment/all_histo_thread_stacked')
        print_i('task completed: got threaded lexical word alignment')
    # Get the overall histogram
    # get_overall_histogram_lexical_word_alignment(alignment_threaded_df)
    print_i('Got lexical word alignment')


def get_semantical_alignment(semantic_preprocessed=True, alignment_ran=True, get_linear=False, get_thread=False):
    """
    Get Semantical alignment
    :param semantic_preprocessed: boolean if data has been preprocessed for this analysis
    :param alignment_ran: boolean if alignment has been ran
    :param get_linear: boolean if we want to investigate the linear threads
    :param get_thread: boolean if we want to investigate the tree-structured threads
    :return: nothing: shows histograms and prints percentiles
    """
    global __threaded_data__, __linear_data__

    # Get alignment of threaded structure
    if get_thread:
        if not semantic_preprocessed:
            # Run semantic preprocessing for thread
            preprocessed_semantic_thread = get_preprocessed_messages_for_semantic(__threaded_data__)
            print(preprocessed_semantic_thread)
            store_data(preprocessed_semantic_thread, __pickle_path_preprocessed_for_semantic_thread__)
        else:
            # Load semantic preprocessing for thread
            preprocessed_semantic_thread = load_data(__pickle_path_preprocessed_for_semantic_thread__)

        if not alignment_ran:
            # Run semantic alignment
            alignment_threaded_df = compute_semantic_alignment(__threaded_data__, preprocessed_semantic_thread, __csv_semantic_alignment_thread__)

        else:
            # Load semantic alignment
            alignment_threaded_df = read_csv(__csv_semantic_alignment_thread__)

        print('threaded', alignment_threaded_df)

        get_overall_histogram_semantic_alignment(alignment_threaded_df, './Results/Semantical_alignment/all_histo_thread')

    # Get alignment of linear structure
    if get_linear:
        # run linear semantic alignment
        if not semantic_preprocessed:
            # Run semantic preprocessing for linear
            preprocessed_semantic_linear = get_preprocessed_messages_for_semantic(__linear_data__)
            store_data(preprocessed_semantic_linear, __pickle_path_preprocessed_for_semantic_linear__)
        else:
            # Load semantic preprocessing for linear
            preprocessed_semantic_linear = load_data(__pickle_path_preprocessed_for_semantic_linear__)

        if not alignment_ran:
            # Run semantic alignment
            alignment_linear_df = compute_semantic_alignment(__linear_data__, preprocessed_semantic_linear,
                                                             __csv_semantic_alignment_linear__)
        else:
            # Load semantic alignment
            alignment_linear_df = read_csv(__csv_semantic_alignment_linear__)

        print('linear', alignment_linear_df)

        get_overall_histogram_semantic_alignment(alignment_linear_df,
                                                     './Results/Semantical_alignment/all_histo_linear')


get_preprocessed_data()
# get_preprocessed_data_from_pickle()
get_lexical_word_alignment(lexical_preprocessed=False, alignment_ran=False, get_linear=True, get_thread=True)
# get_syntactical_alignment(semantic_preprocessed=False, alignment_ran=False, get_linear=True, get_thread=True)