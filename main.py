from Helpers import read_csv
from linguistic_alignment_analysis.compute_lexical_word_alignment import get_preprocessed_messages_for_lexical_word, \
    compute_lexical_word_alignment, get_overall_histogram_lexical_word_alignment
from linguistic_alignment_analysis.compute_syntactical_alignment import get_syntactical_alignment
from linguistic_alignment_analysis.preprocessing_all import run_preprocessing
import pickle


__threaded_data__ = {}
__linear_data__ = {}
__pickle_path_preprocessed_from_enough_authors_enough_posts_linear__ = './PickleData/preprocessed_from_enough_authors_enough_posts_linear'
__pickle_path_preprocessed_from_enough_authors_enough_posts_tree__ = './PickleData/preprocessed_from_enough_authors_enough_posts_tree'
__pickle_path_preprocessed_for_lexical_word = './PickleData/preprocessed_lexical_word'
__pickle_path_preprocessed_for_lexical_word_linear = './PickleData/preprocessed_lexical_word_linear'
__csv_lexical_word_alignment_thread__ = './AlignmentData/lexical_word_alignment.csv'
__csv_lexical_word_alignment_linear__ = './AlignmentData/lexical_word_alignment_linear.csv'

"""
Stores the preprocessed data as pickle to work faster
"""
def store_data(data, path):
    print('[TASK] Pickling data to ', path)
    store_file = open(path, 'ab')
    pickle.dump(data, store_file)
    store_file.close()
    print('[INFO] task completed')


"""
Loads the preprocessed data from pickle to work faster
"""
def load_data(path):
    print('[TASK] Loading preprocessed data from pickle path', path)
    store_file = open(path, 'rb')
    discussion = pickle.load(store_file)
    store_file.close()
    print('[INFO] Task completed')
    return discussion


"""
Main function for getting the preprocessed data
"""
def get_preprocessed_data():
    print('[TASK] Preprocessing data')
    global __threaded_data__, __linear_data__
    __threaded_data__, __linear_data__ = run_preprocessing()
    store_data(__threaded_data__, __pickle_path_preprocessed_from_enough_authors_enough_posts_tree__)
    store_data(__linear_data__, __pickle_path_preprocessed_from_enough_authors_enough_posts_linear__)
    print('[INFO] Preprocessing completed')


"""
Main function for getting the preprocessed data from a pickle
"""
def get_preprocessed_data_from_pickle():
    print('[TASK] Getting preprocessed data')
    global __threaded_data__, __linear_data__
    __threaded_data__ = load_data(__pickle_path_preprocessed_from_enough_authors_enough_posts_tree__)
    __linear_data__ = load_data(__pickle_path_preprocessed_from_enough_authors_enough_posts_linear__)
    print('[INFO] Got preprocessed data')


"""
Get lexical word alignment
"""
def get_lexical_word_alignment():
    print('[TASK] Getting lexical word alignment')
    global __threaded_data__, __linear_data__

    # Use this code if not yet pickled: IMPORTANT
    # preprocessed_lexical_word = get_preprocessed_messages_for_lexical_word(__threaded_data__)
    # store_data(preprocessed_lexical_word, __pickle_path_preprocessed_for_lexical_word)

    # preprocessed_lexical_word_linear = get_preprocessed_messages_for_lexical_word(__linear_data__)
    # store_data(preprocessed_lexical_word_linear, __pickle_path_preprocessed_for_lexical_word_linear)

    # Use this code if pickled: IMPORTANT
    # preprocessed_lexical_word = load_data(__pickle_path_preprocessed_for_lexical_word)
    # preprocessed_lexical_word_linear = load_data(__pickle_path_preprocessed_for_lexical_word_linear)
    # print(preprocessed_lexical_word_linear)

    # Use this code if not yet ran alignment: IMPORTANT
    # alignment_threaded_df = compute_lexical_word_alignment(__threaded_data__, preprocessed_lexical_word, __csv_lexical_word_alignment_thread__)
    # alignment_linear_df = compute_lexical_word_alignment(__linear_data__, preprocessed_lexical_word_linear, __csv_lexical_word_alignment_linear__)

    # Use this code if already ran alignment: IMPORTANT
    # alignment_threaded_df = read_csv(__csv_lexical_word_alignment_thread__)
    alignment_linear_df = read_csv(__csv_lexical_word_alignment_linear__)

    # Get the overall histogram
    get_overall_histogram_lexical_word_alignment(alignment_linear_df)
    print('[INFO] Got lexical word alignment')


# get_preprocessed_data()
# get_preprocessed_data_from_pickle()
get_lexical_word_alignment()
# get_syntactical_alignment()
