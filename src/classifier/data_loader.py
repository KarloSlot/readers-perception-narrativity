import numpy as np
import pandas as pd


def load_annotated_data(threshold):
    """
    Loads Annotated Dataset (2 class). Returns X & Y.
    """
    filename = 'Universal_Annotation_Results_Selection.csv'
    try:
        path = '../data/' + filename
        df = pd.read_csv(path)
    except FileNotFoundError:
        path = '../../data/' + filename
        df = pd.read_csv(path)

    print("Loading annotated data from:", path)

    X, Y = [], []
    for fname, score in df[['FILENAME', 'avg_overall']].values:
        if score > threshold:
            Y.append('POS')
        else:
            Y.append('NEG')
        X.append(fname)
    return np.array(X), np.array(Y)
