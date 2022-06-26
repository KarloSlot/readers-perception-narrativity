import config
from os import path
import xml.etree.ElementTree as ET
import string
import itertools
import csv
import pandas as pd
import numpy as np
from nltk.util import ngrams
from collections import defaultdict
from string import punctuation

if path.exists('../data/BookNLP/'):
    BOOK_PATH_1 = '../data/BookNLP/'
    # Uncomment one (out of the 2) when predicting: new poetry-annotations now
    # BOOK_PATH_2 = '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/booknlp-output-science-jstor/'
    BOOK_PATH_2 = '../data/BookNLP/'
    BOOK_PATH_3 = '../data/BookNLP/'
else:
    BOOK_PATH_1 = '../../data/BookNLP/'
    BOOK_PATH_2 = '../../data/BookNLP/'
    BOOK_PATH_3 = '../../data/BookNLP/'

print('\n----\nUsing the BookNLP path:', BOOK_PATH_1, "\n----\n")


def temporal_order(fname):
    """
    Numerator --> # "BEFORE" and "AFTER" TLINKS
    Denominator --> Total number of TLINKS
    
    We use the Tarsqi Toolkit: https://github.com/tarsqi/ttk
    """
    ttk_fname = config.TTK_PATH + 'ttkoutput_' + fname[:-4] + '.xml'
    dic = process_tlinks(ttk_fname)
    num = dic['BEFORE'] + dic['AFTER']
    denom = sum(dic.values())
    if denom == 0:
        return 0
    return float(num) / denom


def temporality(fname):
    """
    Numerator --> # "TIME", "DATE", "DURATION", "SET" tags
    Denominator --> # words (excluding punctuations)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    num = df.loc[df['ner'].isin(['TIME', 'DATE', 'DURATION', 'SET'])].shape[0]
    words = get_words(df)
    frac = num / float(len(words))
    return frac


def saying(fname):
    """
    Numerator --> # words with verb.communication as BookNLP's supersense
    Denominator --> # words (excluding punctuations)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    num = df.loc[df['supersense'].str.contains('verb.communication')].shape[0]
    words = get_words(df)
    frac = num / float(len(words))
    return frac


def setting(fname):
    """
    Numerator --> # "LOCATION" + # noun.artifact supersense + "ORGANIZATION" with pobj + # from a pre-defined PLACES vocabulary (created via hyponym tree traversal)
    Denominator --> # words (excluding punctuations)
    
    Note that we only check for nouns when matching with PLACES lexicon. The lexicon was created using hyponyms.
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    words = get_words(df)

    location = df.loc[df['ner'].isin(['LOCATION'])].shape[0]  # get location tags
    org_places = len(organization_tags(df, out='place'))  # organization with pobj tags

    # places-vocabulary (we only check for nouns)
    vocab = place_lexicon_count(df.loc[df['pos'].str.startswith('N')]['word'].tolist())

    artifacts = df.loc[df['supersense'].str.contains('noun.artifact')].shape[0]
    #     print("Location Tag:", location, "| ORGANIZATION-Places:", org_places, "| Places-Vocabulary:", vocab, "| Artifacts:", artifacts, "| Words:", len(words))
    frac = (location + artifacts + org_places + vocab) / float(len(words))
    return frac


def concreteness(fname):
    """
    Returns a ratio for concreteness:
    Numerator --> sum of concreteness scores of every word/bigram in the text that appears in Brysbaert et al's lexicon.
    Denominator -->  # words (excluding punctuations)
    
    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    concreteness score (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    word_list = get_words(df)

    conc_score = 0.0
    for n in [1, 2]:
        for tup in ngrams(word_list, n):
            temp = ''  # temp is the unigram/bigram/trigram
            for word in tup:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in config.MAP_WORD_CONC:  # check if it exists in lexicon
                conc_score += config.MAP_WORD_CONC[temp]
    #                 print("Exists:", temp, "| Score:", config.MAP_WORD_CONC[temp])
    return conc_score / len(word_list)


def eventfulness(fname):
    """
    The rate of all non-helping and non-stative verbs in a passage relative to the number of words.
    Numerator --> # verbs (non-helping and non-stative)
    Denominator --> # words (excluding punctuations)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    words = get_words(df)
    # Count verbs:
    verbs = 0
    verb_df = df.loc[df['pos'].str.startswith('VB')]
    for word, supersense in verb_df[['word', 'supersense']].values:
        if word.lower() in config.HELPING_VERBS or 'verb.stative' in supersense:  # skip helping & stative verbs
            #             print("Skip:", word)
            continue
        verbs += 1
    frac = verbs / float(len(words))
    return frac


def feltness(fname):
    """
    Measured as the rate of perceptual, cognitive, or emotion terms over the total number of words.
    
    Numerator --> # words with noun.cognition or verb.cognition or noun.feeling or verb.emotion or verb.perception as BookNLP's supersense
    Denominator --> # words (excluding punctuations)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    feltness_categories = "noun.cognition|verb.cognition|noun.feeling|verb.emotion|verb.perception"
    num = df.loc[df['supersense'].str.contains(feltness_categories)].shape[0]
    words = get_words(df)
    frac = num / float(len(words))
    return frac


def coherence(fname, kind):
    """
    Word-Overlap model (similar to Coh-Metrix's "Referential Cohesion"): computes the number of tokens that overlap in sentence 1 and sentence 2.
    Only looks at relevant words -- nouns, verbs, adverbs, adjectives and pronouns (via coreference resolution).
    
    
    Parameters
    ----------
    fname: filename for BookNLP output
    kind: "seq" or "global"
            seq looks at sequential pairs of sentences: 01, 12, 23, 34
            global looks at all possible pairs of sentences: 01, 02, 03, 04, 12, 13, ..., 34

    Returns
    -------
    float: average coherence score for the pairs considered
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    num_sents = len(set(df['sentenceID'].tolist()))
    ratios = []
    for id1, id2 in itertools.combinations(list(range(0, num_sents)), 2):
        if kind == 'seq':
            if id2 != id1 + 1:
                continue
        sentdf_1 = df.loc[df['sentenceID'] == id1][['word', 'pos', 'ner', 'characterId']]
        sentdf_2 = df.loc[df['sentenceID'] == id2][['word', 'pos', 'ner', 'characterId']]
        overlap_ratio = word_overlap(sentdf_1, sentdf_2)
        ratios.append(overlap_ratio)

    if len(ratios) == 0:
        return -100
    return np.array(ratios).mean()


def agency(fname):
    """
    Agency is computed as the mean distance between a sentence's subject and the most immediate verb measured in tokens.
    Average for all 'nsubj' in the passage.
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    n_s = []
    for token_id, word, pos, dep in df[['tokenId', 'word', 'pos', 'dependency_relation']].values:
        if dep == 'nsubj':  # find the next immediate verb
            n = immediate_verb(token_id, df)
            #             print(word, '| Distance:', n)
            n_s.append(n)

    if len(n_s) == 0:
        #         print(fname, "has no nsubj.")
        return 67.0  # largest agency value in training-data (itself an outlier - mean is around 3)

    mean_distance = np.array(n_s).mean()
    return mean_distance


def agenthood(fname):
    """
    Modeled as the presence of an "animate entity" in the subject position of a sentence.
    Animate entities are detected using a combination of NER, Supersense, LitBank tag, and a classifier trained on Jahan et al's data using Karsdosp et al's approach.
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    map_tokenid_pred = {}
    for token_id, word, supersense, ner in df[['tokenId', 'word', 'supersense', 'ner']].values:
        pred = predict_animacy_lite(fname, token_id, word, supersense, ner)
        map_tokenid_pred[token_id] = pred

    df['animacy_prediction'] = df['tokenId'].map(map_tokenid_pred)

    num = df.loc[(df['animacy_prediction'] == 'animate') & (df['dependency_relation'].str.contains('subj'))].shape[0]
    words = get_words(df)
    frac = num / float(len(words))
    return frac


def predict_animacy_lite(fname, token_id, word, supersense, ner):
    """
    Lite-Version:
    Word is predicted to be animate if one or more out of the following methods consider it animate.
    1) BookNLP Supersense 'person/animal', 2) NER 'PERSON' tag
    Note that all pronouns (from config.py) are classified as animate.
    """
    if word.lower() in config.PRONOUNS:
        return 'animate'

    if "noun.person" in supersense or "noun.animal" in supersense:
        S = 1
    else:
        S = 0

    if ner == 'PERSON':
        N = 1
    else:
        N = 0

    score = S + N
    if score >= 1:
        return "animate"
    else:
        return "inanimate"


# KARS_DF = pd.read_csv(config.KARS_PATH, delimiter='\t')
# def karsdorp_prediction(filename, token_id, word):
#     """
#     Returns the prediction for the given filename and tokenID.
#     """
#     row = KARS_DF.loc[(KARS_DF['filename']==filename) & (KARS_DF['tokenid']==token_id)]
#     if row.empty:
#         return 'inanimate'
#     assert row['word'].values[0] == word
#     return 'animate'

# def is_litbank_person(fname, ID):
#     """
#     Returns True if the given ID is tagged "PER" by LitBank.
#     """
#     token_ids = [] # list of token IDs tagged as PER by LitBank tagger
#     with open(config.LITBANK_PATH+fname, 'r') as f:
#         for line in f.readlines():
#             id1, id2, tag, word = line.split('\t')
#             id1 = int(id1)
#             id2 = int(id2)
#             if tag == 'PER':
#                 for i in list(range(id1, id2+1)):
#                     token_ids.append(int(i))
#     token_ids = list(set(token_ids))
#     return ID in token_ids


def place_lexicon_count(word_list):
    """
    Counts the number of words/bigrams/trigrams present in the input list of words "word_list".
    Compares against the PLACES Lexicon which was created using hyponyms.

    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    int
        Count of the number of places-words
    """
    count = 0
    for n in [1, 2, 3]:
        for tup in ngrams(word_list, n):
            temp = ''  # temp is the unigram/bigram/trigram
            for word in tup:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in config.PLACES_VOCAB:  # check if it exists in lexicon
                count += 1
    return count


def organization_tags(df, out='agent'):
    """
    - Process all entities tagged "ORGANIZATION" by BookNLP. It chunks two adjacent tags as one entity.
    - If entity is in a "pobj" relation, it's treated as a place/setting.
    - If "pobj" is not present, it's treated as an agent.
    
    out can be 'agent' or 'place'. Returns the list specified by out.
    """
    assert out in ['agent', 'place']
    continuous_chunk_setting = []  # for "pobj" instances
    continuous_chunk_agent = []  # when "pobj" is not present
    current_chunk, deprels = [], []

    for token, tag, dep in df[['word', 'ner', 'dependency_relation']].values:
        if tag == "ORGANIZATION":
            current_chunk.append(token)
            deprels.append(dep)
        else:
            if current_chunk:  # if the current chunk is not empty
                if "pobj" in deprels:
                    continuous_chunk_setting.append(' '.join(current_chunk))
                else:
                    continuous_chunk_agent.append(' '.join(current_chunk))
                current_chunk = []
                deprels = []

    if current_chunk:  # # flush the final current_chunk into the continuous_chunk, if any
        if "pobj" in deprels:
            continuous_chunk_setting.append(' '.join(current_chunk))
        else:
            continuous_chunk_agent.append(' '.join(current_chunk))
        current_chunk = []
        deprels = []

    places = [''.join(chunk) for chunk in continuous_chunk_setting]
    agents = [''.join(chunk) for chunk in continuous_chunk_agent]
    #     print("Places: {} | Agents: {}".format(places, agents))
    if out == 'agent':
        return agents
    elif out == 'place':
        return places


def filter_punct(tokens):
    """
    Given a list of all tokens, returns a list of words. Punctuations are skipped.
    """
    words = []
    for token in tokens:
        if str(token) not in string.punctuation:
            words.append(token)
    return words


def immediate_verb(token_id, df):
    """
    Return the number of tokens between the given 'nsubj' tokenID and the most immediate verb in the BookNLP dataframe.
    Both subj and verb included in the count.
    """
    df = df.loc[df['tokenId'] >= token_id]

    c = 0
    for word, pos in df[['word', 'pos']].values:
        c += 1
        if pos.startswith('VB'):
            break
    return c


def get_words(df):
    """
    Given the BookNLP dataframe, returns a list of words. Punctuations are skipped.
    """
    df = df.loc[df['dependency_relation'] != 'punct']  # remove punctuations
    words = filter_punct(df['word'].tolist())  # filter punctuations again in case BookNLP missed any
    return words


def get_POS_str(fname):
    """
    Returns a string for all part-of-speech tags in the given filename.
    """
    try:
        df = pd.read_csv(BOOK_PATH_1 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    except:
        try:
            df = pd.read_csv(BOOK_PATH_2 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
        except:
            df = pd.read_csv(BOOK_PATH_3 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['POS_tag'].tolist())


def get_dep_str(fname):
    """
    Returns a string for all dependency tags in the given filename.
    """
    try:
        df = pd.read_csv(BOOK_PATH_1 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    except:
        df = pd.read_csv(BOOK_PATH_2 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['dependency_relation'].tolist())


def get_words_str(fname):
    """
    Returns a string for all 'word' tokens in the given BookNLP DataFrame.
    """
    try:
        df = pd.read_csv(BOOK_PATH_1 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    except:
        df = pd.read_csv(BOOK_PATH_2 + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['word'].tolist())


def process_tlinks(fname):
    """
    Given the filename of TTK's output xml, it returns a dictionary for all TLINK relType(s).
    """
    root = ET.parse(fname).getroot()
    c = defaultdict(int)
    for t in root.findall('tarsqi_tags/'):
        if t.tag == 'TLINK':
            instance = t.attrib
            c[instance['relType']] += 1
    return c


def get_character_ids(sent_df):
    """
    Returns a list of unique character IDs present in sent_df.
    """
    return list(set(sent_df['characterId'].tolist()))


def word_overlap(sentdf_1, sentdf_2):
    """
    Computes the number of tokens that overlap in sentence 1 and sentence 2.
    Iterates over tokens in the first sentence: +1 if it appears in sentence 2 as well. +1 if its coreferent appears in the second sentence.    
    Only looks at relevant words -- nouns, verbs, adverbs, adjectives and pronouns (via coreference resolution).
    When char_id for a PERSON is -1 (BookNLP failed at coreference resolution), we consider it an overlap if there is a pronoun in the next sentence.

    Input: DataFrame for sentence 1 and sentence 2
    Returns: ratio of number of overlapping words to total number of relevant words in sentence 1
    """
    sent2_words = [w.lower() for w in sentdf_2['word'].tolist()]
    sent2_chars = get_character_ids(sentdf_2)

    total_words_considered, overlap = [], []

    prev_ner = False  # True if the previous word was tagged NER

    for word, pos, ner, char_id in sentdf_1.values:
        #         print("Sent2 words considered:", len(sent2_words))
        if char_id == -1:  # check against list of words (no BookNLP coreference-resolution)
            if ner == 'PERSON' and not prev_ner:  # we match this with a pronoun in sent2; counts a chunked person-entity as one
                total_words_considered.append(word)
                for w in sent2_words:
                    if w in config.PRONOUNS:
                        #                         print("{} matches with {} in second sentence.".format(word, w))
                        overlap.append(word)
                        sent2_words.remove(w)  # remove so we don't double count the same pronoun
                        break

            elif pos.startswith('NN') or pos.startswith('RB') or pos.startswith('VB') or pos.startswith(
                    'JJ') or word.lower() in config.PRONOUNS:
                total_words_considered.append(word)
                if word.lower() in sent2_words:
                    overlap.append(word)


        else:  # check for coreference-resolution
            total_words_considered.append(word)
            #             print("Processing character:", word, "| ID:", char_id)
            if char_id in sent2_chars:
                #                 print("Present in the second sentence:", word, sent2_chars)
                overlap.append(word)
            elif word.lower() in sent2_words:
                overlap.append(word)

        if ner == 'PERSON':
            prev_ner = True
        else:
            prev_ner = False

    #     print("Overlap:", overlap)
    #     print("Total sent 1 words considered:", total_words_considered)
    if len(total_words_considered) == 0: return 0
    return len(overlap) / float(len(total_words_considered))


def filter_punct(tokens):
    """
    Removes all punctuations and returns a list of words.
    """
    return [word for word in tokens if word not in punctuation]


def compute_quoted_words(fname):
    """
    Returns fraction of number of quoted words to total number of words.
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    # Remove punctuations and B-QUOTE's:
    temp = df.loc[df['inQuotation'] != 'B-QUOTE']
    df_no_punct = temp.loc[temp['dependency_relation'] != 'punct']
    all_words = filter_punct(df_no_punct['word'].tolist())
    quoted_words = filter_punct(df_no_punct.loc[df_no_punct['inQuotation'] == 'I-QUOTE'][
                                    'word'].tolist())  # filter punctuations again incase BookNLP missed any
    return len(quoted_words) / len(all_words)
