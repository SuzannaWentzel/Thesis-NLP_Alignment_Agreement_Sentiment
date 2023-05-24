import pandas as pd

"""
Reads discussions into pandas
"""
def read_csv(path):
    print('[TASK] reading csv from: ', path)
    df = pd.read_csv(path)
    print('[RESULT]', df)
    print('[INFO] task completed')
    return df


def color_scheme():
    return ['#d74a94', '#6bb2a5', '#fdd516', '#77b75b', '#ff8800']