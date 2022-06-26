import pandas as pd
import numpy as np
import os

def load_annotated_data():
    """
    Loads Annotated Dataset (2 class). Returns X & Y.
    """

    # Data from Piper 2022
    path = 'MinNarrative_ReaderData_Final.csv'
    df = pd.read_csv(path)
    print("Loading annotated data from:", path)

    X, Y = [], []
    for fname in df['FILENAME'].values:
        X.append(fname)
    return np.array(X)

annotated_files = load_annotated_data()
print(len(annotated_files))
for subdir, dirs, files in os.walk("minNarrative_txtfiles"):
    for file in files:
        print(file)
        if file not in annotated_files:
            filepath = os.path.join(subdir, file)
            # annotated_files = np.delete(annotated_files, np.where(annotated_files == file))
            os.remove(filepath)

# print(annotated_files)