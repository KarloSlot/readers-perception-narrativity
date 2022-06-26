import features
import config

import pandas as pd
import pickle
import os
import csv

# main_df = pd.read_csv('/Users/sunyambagga/Desktop/txtLAB-2/minimal-narrativity/data/dataset_17_May_2021.tsv', delimiter='\t')
# fnames = main_df['FILENAME'].tolist()

# folder_name = 'POETRY' # 'SCIENCE-ROYAL' or 'SCIENCE-JSTOR'
# path = '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/'+folder_name

folder_name = 'poetry'
path = '/Users/sunyambagga/Desktop/txtLAB-2/detecting-narrativity/data/hand-annotate/' + folder_name

fnames = os.listdir(path)
print("\n\n-------\nProcessing:", folder_name)

map_fname_featuredict = {}
c = 0
no_nsubj = 0
for fname in fnames:
    #     print("Processing:", fname)
    try:
        df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    except:
        print("BookNLP output not available for:", fname, "| Skip.")
        continue

    feature_dict = {}
    feature_dict['temporality'] = features.temporality(fname)
    #     feature_dict['temporal_order'] = features.temporal_order(fname)

    feature_dict['setting'] = features.setting(fname)
    feature_dict['concreteness'] = features.concreteness(fname)
    feature_dict['saying'] = features.saying(fname)
    feature_dict['eventfulness'] = features.eventfulness(fname)

    feature_dict['agenthood'] = features.agenthood(fname)
    ag = features.agency(fname)
    if ag == 67:
        no_nsubj += 1
    feature_dict['agency'] = ag
    feature_dict['coh_seq'] = features.coherence(fname, 'seq')
    #     feature_dict['coh_global'] = features.coherence(fname, 'global')
    feature_dict['feltness'] = features.feltness(fname)
    feature_dict['pct_quoted'] = features.compute_quoted_words(fname)

    map_fname_featuredict[fname] = feature_dict

    if c % 1000 == 0:
        print("Done with:", c)
    c += 1

print("\nNote {} passages in {} have agency = 67 (no nsubjs possibly)".format(no_nsubj, folder_name))
with open('../../pickles/' + folder_name + '_tense_mood_voice_features_lite_handannotate.pickle', 'wb') as f:
    pickle.dump(map_fname_featuredict, f)
