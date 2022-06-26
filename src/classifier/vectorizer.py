import features
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

p = '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/'
# with open(p+'pickles/tmv_features_lite_merged.pickle', 'rb') as f:
#     TMV_FEATURES_TRAIN = pickle.load(f) # created via pickle_features.py
# with open(p+'pickles/poetry_tense_mood_voice_features_lite_handannotate.pickle', 'rb') as f:
#     TMV_FEATURES_NEWPOETRY = pickle.load(f) # created via pickle_features.py
# with open(p+'pickles/POETRY_tense_mood_voice_features_lite.pickle', 'rb') as f:
#     TMV_FEATURES_POETRY = pickle.load(f) # created via pickle_features.py
# with open(p+'pickles/SCIENCE-JSTOR_tense_mood_voice_features_lite.pickle', 'rb') as f:
#     TMV_FEATURES_SCIENCE1 = pickle.load(f) # created via pickle_features.py
# with open(p+'pickles/SCIENCE-ROYAL_tense_mood_voice_features_lite.pickle', 'rb') as f:
#     TMV_FEATURES_SCIENCE2 = pickle.load(f) # created via pickle_features.py

# TMV_FEATURES = {**TMV_FEATURES_TRAIN, **TMV_FEATURES_NEWPOETRY, **TMV_FEATURES_POETRY, **TMV_FEATURES_SCIENCE1, **TMV_FEATURES_SCIENCE2}
# print("Loading TMV features pickle... Size:", len(TMV_FEATURES))

TMV_DF_file = 'Universal_Annotation_Results_Selection.csv'
try:
    TMV_DF = pd.read_csv('../data/' + TMV_DF_file)
except FileNotFoundError:
    TMV_DF = pd.read_csv('../../data/' + TMV_DF_file)
MAX_FEATURES = None


def all_feature_categories(train_x, test_x):
    """
    Vectorizes the input text using all feature categories:
    pos1 (max=100) + word1 (max=5000) + dep23 (max=5000) + tense + mood + voice

    We pick the top-performing pos, word, and dep features.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    word_train, word_test = word_unigrams(train_x, test_x, maxfeat=5000)
    dep_train, dep_test = dep_bitri_grams(train_x, test_x, maxfeat=5000)

    tmv_train, tmv_test = tense_mood_voice(train_x, test_x)
    combined_train = np.hstack((tmv_train, dep_train.toarray(), pos_train.toarray(), word_train.toarray()))
    combined_test = np.hstack((tmv_test, dep_test.toarray(), pos_test.toarray(), word_test.toarray()))

    print("Train -- tmv: {} | dep: {} | pos: {} | word: {}".format(tmv_train.shape, dep_train.shape, pos_train.shape,
                                                                   word_train.shape))
    print("Test -- tmv: {} | dep: {} | pos: {} | word: {}".format(tmv_test.shape, dep_test.shape, pos_test.shape,
                                                                  word_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def all_feature_categories_uni(train_x, test_x):
    """
    Vectorizes the input text using all feature categories (simpler model - unigrams only & max=100):
    pos1 (max=100) + word1 (max=100) + dep1 (max=100) + tense + mood + voice

    We pick the top-performing pos, word, and dep features.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    word_train, word_test = word_unigrams(train_x, test_x, maxfeat=100)
    dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)

    tmvq_train, tmvq_test = tense_mood_voice_quoted(train_x, test_x)
    combined_train = np.hstack((tmvq_train, dep_train.toarray(), pos_train.toarray(), word_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((tmvq_test, dep_test.toarray(), pos_test.toarray(), word_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- tmv-quoted: {} | dep: {} | pos: {} | word: {}".format(tmvq_train.shape, dep_train.shape,
                                                                          pos_train.shape, word_train.shape))
    print(
        "Test -- tmv-quoted: {} | dep: {} | pos: {} | word: {}".format(tmvq_test.shape, dep_test.shape, pos_test.shape,
                                                                       word_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using part-of-speech unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)

    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()

    else:
        return X_train, X_test


def pos_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))

    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()
    else:
        return X_train, X_test


def dep_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def tense2(train_x, test_x):
    """
    Vectorizes the input text using tense features: [temporality, temporal_order]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([TMV_FEATURES[fname]['temporality']])  # , TMV_FEATURES[fname]['temporal_order']])
    for fname in test_x:
        X_test.append([TMV_FEATURES[fname]['temporality']])  # , TMV_FEATURES[fname]['temporal_order']])
    return np.array(X_train), np.array(X_test)


def tense(train_x, test_x):
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['temporality'].to_string(
            index=False))])  # , TMV_FEATURES[fname]['temporal_order']])
    for fname in test_x:
        X_test.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['temporality'].to_string(
            index=False))])  # , TMV_FEATURES[fname]['temporal_order']])
    return np.array(X_train), np.array(X_test)


def mood2(train_x, test_x):
    """
    Vectorizes the input text using mood features: [setting, concreteness, saying, eventfulness]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append(
            [TMV_FEATURES[fname]['setting'], TMV_FEATURES[fname]['concreteness'], TMV_FEATURES[fname]['saying'],
             TMV_FEATURES[fname]['eventfulness']])
    for fname in test_x:
        X_test.append(
            [TMV_FEATURES[fname]['setting'], TMV_FEATURES[fname]['concreteness'], TMV_FEATURES[fname]['saying'],
             TMV_FEATURES[fname]['eventfulness']])
    return np.array(X_train), np.array(X_test)


def voice2(train_x, test_x, coh_kind='seq'):
    """
    Vectorizes the input text using mood features: [agenthood, agency, coherence, feltness]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append(
            [TMV_FEATURES[fname]['agenthood'], TMV_FEATURES[fname]['agency'], TMV_FEATURES[fname]['coh_' + coh_kind],
             TMV_FEATURES[fname]['feltness']])
    for fname in test_x:
        X_test.append(
            [TMV_FEATURES[fname]['agenthood'], TMV_FEATURES[fname]['agency'], TMV_FEATURES[fname]['coh_' + coh_kind],
             TMV_FEATURES[fname]['feltness']])
    return np.array(X_train), np.array(X_test)


def voice(train_x, test_x, coh_kind='seq'):
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['agenthood']),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['agency']),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['coh_' + coh_kind]),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['feltness'])])
    for fname in test_x:
        X_test.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['agenthood']),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['agency']),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['coh_' + coh_kind]),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['feltness'])])
    return np.array(X_train), np.array(X_test)


def mood(train_x, test_x):
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['setting']),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['concreteness']),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['saying']),
                        float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['eventfulness'])])
    for fname in test_x:
        X_test.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['setting']),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['concreteness']),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['saying']),
                       float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['eventfulness'])])
    return np.array(X_train), np.array(X_test)


def pct_quoted2(train_x, test_x):
    """
    Vectorizes the input text using quoted-feature.
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([TMV_FEATURES[fname]['pct_quoted']])
    for fname in test_x:
        X_test.append([TMV_FEATURES[fname]['pct_quoted']])
    return np.array(X_train), np.array(X_test)


def pct_quoted(train_x, test_x):
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['pct_quoted'])])
    for fname in test_x:
        X_test.append([float(TMV_DF.loc[TMV_DF['FILENAME'] == fname]['pct_quoted'])])
    return np.array(X_train), np.array(X_test)


def tense_mood_voice(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using all 10 tense/mood/voice features:
    [temporality, temporal_order, setting, concreteness, saying, eventfulness, agenthood, agency, coherence, feltness]

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    tense_train, tense_test = tense(train_x, test_x)
    mood_train, mood_test = mood(train_x, test_x)
    voice_train, voice_test = voice(train_x, test_x)
    combined_train = np.hstack([tense_train, mood_train, voice_train])
    combined_test = np.hstack([tense_test, mood_test, voice_test])

    #     feat_names = ['temporality', 'temporal_order', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood', 'agency', 'coherence', 'feltness']
    feat_names = ['temporality', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood', 'agency',
                  'coherence', 'feltness']

    if return_feature_names:
        return np.array(combined_train), np.array(combined_test), feat_names
    else:
        return np.array(combined_train), np.array(combined_test)


def tense_mood_voice_quoted(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using all 10 tense/mood/voice features:
    [temporality, temporal_order, setting, concreteness, saying, eventfulness, agenthood, agency, coherence, feltness, quoted]

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    tense_train, tense_test = tense(train_x, test_x)
    mood_train, mood_test = mood(train_x, test_x)
    voice_train, voice_test = voice(train_x, test_x)
    quoted_train, quoted_test = pct_quoted(train_x, test_x)
    combined_train = np.hstack([tense_train, mood_train, voice_train, quoted_train])
    combined_test = np.hstack([tense_test, mood_test, voice_test, quoted_test])

    feat_names = ['temporality', 'temporal_order', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood',
                  'agency', 'coherence', 'feltness', 'pct_quoted']

    if return_feature_names:
        return np.array(combined_train), np.array(combined_test), feat_names
    else:
        return np.array(combined_train), np.array(combined_test)


def word_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names(), vectorizer
    else:
        return X_train, X_test


def word_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_tense(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + tense


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = tense(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_mood(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + mood


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = mood(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_voice(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + voice


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = voice(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_tmv(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + other_funct (tense/mood/voice)


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats
    return combined_train, combined_test


def pos_tmv_quoted(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + TMV + quoted


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice_quoted(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice_quoted(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats
    return combined_train, combined_test


def pos_dep_tmv(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using:
    pos1 (max=100) + dep1 (max=100) + tense/mood/voice

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        dep_train, dep_test, dep_feats = dep_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray(), dep_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray(), dep_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- tmv: {} | pos: {} | dep: {}".format(other_train.shape, pos_train.shape, dep_train.shape))
    print("Test -- tmv: {} | pos: {} | dep: {}".format(other_test.shape, pos_test.shape, dep_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats + dep_feats
    else:
        return combined_train, combined_test
