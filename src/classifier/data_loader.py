import pandas as pd
import numpy as np
import os


# def get_annotated_filenames():
#     """
#     Returns the reader-annotated filenames.
#     """
#     df1 = pd.read_csv(
#         '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/Reader_annotated_data_1/minNarrative_Annotated_Passages_UPDATED.csv')
#     df2 = pd.read_csv('/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/novel19c_105passages.tsv',
#                       delimiter='\t')
#     df3 = pd.read_csv(
#         '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/450passages_poetry_nonfic_science.tsv',
#         delimiter='\t')
#     reader_annotated_fnames = df1['FILENAME'].tolist() + df2['FILENAME'].tolist() + df3['FILENAME'].tolist()
#     print(df1.shape, df2.shape, df3.shape, "| Total reader-annotated files:", len(reader_annotated_fnames))
#     return reader_annotated_fnames
#
#
# def get_mispredictions():
#     """
#     Returns the mispredictions from round1 of training (without Opinions).
#     """
#     fname = 'POS-TMV_13438_predictions.tsv'
#     df = pd.read_csv('/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/results/' + fname, delimiter='\t')
#     mispreds = df.loc[df['Predicted-Label'] != df['True-Label']]['Filename'].tolist()
#     print("Total Mispredictions:", len(mispreds), "| From:", fname)
#     return mispreds
#
#
# def load_data(discard_genres=['OPINION'], remove_annotated_passages=True, remove_mispreds=False):
#     """
#     Loads the dataset.
#     Discards the genres listed in 'discard_genres'
#
#     Returns X and Y
#
#     POSITIVE:
#     Flash Fiction
#     Aesop's Fables
#     Short Story Anthology
#     World Fairy Tales
#     ROCStories
#     NewsHeadlines
#     RedditStories
#
#     NEGATIVE:
#     Aphorisms
#     Academic Discourse (Science & Litstudy)
#     Prayers
#     Bartlett's Quotations
#     Twitter
#     Book Reviews
#     Opinions (CNN & Fox)
#     """
#     BOOK_PATH = '/Users/sunyambagga/Desktop/txtLAB-2/minimal-narrativity/booknlp-output-narrativity/'
#
#     main_df = pd.read_csv('/Users/sunyambagga/Desktop/txtLAB-2/minimal-narrativity/data/dataset_17_May_2021.tsv',
#                           delimiter='\t')
#     main_df = main_df.loc[main_df['NARRATIVITY'].isin(['POS', 'NEG'])]
#     print("# genres initially:", main_df['GENRE'].nunique())
#     main_df = main_df.loc[~main_df['GENRE'].isin(discard_genres)]
#     print("Post-filtering, # genres:", main_df['GENRE'].nunique())
#
#     if remove_annotated_passages:
#         annotated_fnames = get_annotated_filenames()
#         main_df = main_df.loc[~main_df['FILENAME'].isin(annotated_fnames)]
#
#     if remove_mispreds:
#         mispreds = get_mispredictions()
#         main_df = main_df.loc[~main_df['FILENAME'].isin(mispreds)]
#
#     print("Dataset size:", main_df.shape)
#
#     X, Y = [], []
#     for row in main_df.values:
#         narr, genre, kind, fname, text = row
#         if not os.path.exists(BOOK_PATH + fname + '/' + fname + '.tokens'):
#             print("Skip file:", fname)
#             continue
#
#         X.append(fname)
#         Y.append(narr)
#     return np.array(X), np.array(Y)
#
#
# def reader_annotated_dict():
#     """
#     Returns a dictionary mapping each annotated filename to avg-reader-score.
#     """
#     df = pd.read_csv(
#         '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/reader_annotated_with_predictions_416.tsv',
#         delimiter='\t')
#     return dict(df[['FILENAME', 'Avg_Reader_Score']].values)


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


# def load_annotated_data_3class():
#     """
#     Loads Annotated Dataset (3 class). Returns X & Y.
#     (1-2) = NEG
#     [2-3] = NEUTRAL
#     (3,5] = POS
#     """
#     df = pd.read_csv(
#         '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/reader_annotated_with_predictions_416.tsv',
#         delimiter='\t')
#
#     # 3 class where: are the three classes.
#     X, Y = [], []
#     for fname, score in df[['FILENAME', 'Avg_Reader_Score']].values:
#         if score < 2:
#             Y.append('NEG')
#         elif score >= 2 and score <= 3:
#             Y.append('NEUTRAL')
#         elif score > 3:
#             Y.append('POS')
#         X.append(fname)
#     return np.array(X), np.array(Y)
