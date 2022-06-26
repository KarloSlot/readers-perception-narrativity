## Predict on Experimental Data ##


import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append('./classifier/')
import data_loader
import vectorizer
import pandas as pd
import os
import random
import numpy as np
import eli5

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def get_prob_narr(test_fnames):
    """
    Returns a dictionary mapping test-filenames to a probability-narrative.
    """
    if model == 'within-401':  # trained on Reader-Annotated data
        train_fnames, Y = data_loader.load_annotated_data(threshold=2.5)
        print("Using Annotated-Data only..", len(Y))
        X_train, X_test = vectorizer.all_feature_categories_uni(train_fnames, test_fnames)
        algo = RandomForestClassifier(n_estimators=500, max_depth=20,
                                      random_state=seed_value)  # the best pos-TMV parameters

    elif model == 'within-401-pos-tense':
        train_fnames, Y = data_loader.load_annotated_data(threshold=2.5)
        print("Using Annotated-Data only..", len(Y))
        X_train, X_test = vectorizer.pos_tense(train_fnames, test_fnames)
        algo = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=seed_value)

    elif model == 'within-401-pos-mood':
        train_fnames, Y = data_loader.load_annotated_data(threshold=2.5)
        print("Using Annotated-Data only..", len(Y))
        X_train, X_test = vectorizer.pos_mood(train_fnames, test_fnames)
        algo = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=seed_value)

    elif model == 'within-401-pos-tmv-quoted':
        train_fnames, Y = data_loader.load_annotated_data(threshold=2.5)
        print("Using Annotated-Data only..", len(Y))
        X_train, X_test = vectorizer.pos_tmv_quoted(train_fnames, test_fnames)
        algo = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=seed_value)

    elif model == 'word1':
        train_fnames, Y = data_loader.load_annotated_data(threshold=2.5)
        print("Using Annotated-Data only..", len(Y))
        X_train, X_test, names, v = vectorizer.word_unigrams(train_fnames, test_fnames, return_feature_names=True)
        algo = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=seed_value)

    print("Train files:", len(train_fnames), X_train.shape, len(Y), "| Test files:", len(test_fnames), X_test.shape)

    algo.fit(X_train, Y)

    pred_probs = algo.predict_proba(X_test)
    preds = algo.predict(X_test)

    map_fname_probnarr = {}
    for fname, probs, pred in zip(test_fnames, pred_probs, preds):
        prob_narr = probs[1]  # second element (['NEG', 'POS'])
        if prob_narr > 0.5:
            assert pred == 'POS'
        else:
            assert pred == 'NEG'
        map_fname_probnarr[fname] = [prob_narr, pred]

    print("Ordering:", algo.classes_.tolist(), "| Predictions:", len(pred_probs), len(preds))
    return map_fname_probnarr, algo, v, names


def eli5_passage_results(df_piper_2022, df_universals, passage, label, path, algo, vec, names):
    """
    Get the ELI5 analysis of a specific passage

    :param df_piper_2022: DataFrame Object of data set from Piper (2022)
    :param df_universals: DataFrame Object of data set from our experiment
    :param passage: DataFrame Object with single passage to get information of
    :param label: String representing directory name
    :param path: Path to put results
    :param algo: Algorithm object used
    :param vec: Vectorizer used
    :param names: Feature names
    """
    new_path = path + label + '/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    passage_piper_2022 = df_piper_2022.loc[df_piper_2022['FILENAME'] == passage['FILENAME'].to_list()[0]]
    passage_universals = df_universals.loc[df_universals['FILENAME'] == passage['FILENAME'].to_list()[0]]
    with open(new_path + 'results.txt', 'w') as f:
        f.write('Filename:                             {0}\n'.format(passage['FILENAME'].to_list()[0]))
        f.write('Piper (2022) - Probability.Narrative: {0}\n'.format(
            passage_piper_2022['Probability.Narrative'].to_list()[0]))
        f.write('Piper (2022) - Agency scores:         {0}\n'.format(passage_piper_2022[['AY_agency', 'ML_agency',
                                                                                         'LM_agency']].values[0]))
        f.write('Piper (2022) - Event scores:          {0}\n'.format(passage_piper_2022[['AY_event', 'ML_event',
                                                                                         'LM_event']].values[0]))
        f.write('Piper (2022) - World scores:          {0}\n'.format(passage_piper_2022[['AY_world', 'ML_world',
                                                                                         'LM_world']].values[0]))
        f.write('Universals - Probability.Narrative:   {0}\n'.format(
            passage_universals['Probability.Narrative'].to_list()[0]))
        f.write('Universals - Suspense scores:         {0}\n'.format(passage_universals[['suspense_cmb_0',
                                                                                         'suspense_cmb_1',
                                                                                         'suspense_cmb_2']].values[
                                                                         0].astype(int)))
        f.write('Universals - Curiosity scores:        {0}\n'.format(passage_universals[['curiosity_cmb_0',
                                                                                         'curiosity_cmb_1',
                                                                                         'curiosity_cmb_2']].values[
                                                                         0].astype(int)))
        f.write('Universals - Surprise scores:         {0}\n'.format(passage_universals[['surprise_cmb_0',
                                                                                         'surprise_cmb_1',
                                                                                         'surprise_cmb_2']].values[
                                                                         0].astype(int)))
    text = passage['TEXT'].to_list()[0]
    with open(new_path + "eli5_prediction.html", "w") as file:
        file.write(eli5.formatters.html.format_as_html(eli5.explain_prediction(algo, text, vec=vec,
                                                                               feature_names=names,
                                                                               targets=['POS'])))


def main():
    """
    Saves the narrative-probabilities to dataset CSV.
    """
    csv_filename = '../data/Universal_Annotation_Results_Selection.csv'
    df = pd.read_csv(csv_filename)
    test_fnames = list(df['FILENAME'].values)

    map_fname_probnarr, algo, vec, names = get_prob_narr(test_fnames)

    print("\n\nWrite to Dataset..")

    for k, v in map_fname_probnarr.items():
        df.loc[df.FILENAME == k, "Probability.Narrative"] = v[0]
        df.loc[df.FILENAME == k, "NARRATIVITY"] = v[1]
    df.to_csv(csv_filename, index=False)

    results_path = '../results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    eli5_results_path = '../results/eli5_results/'
    if not os.path.exists(eli5_results_path):
        os.mkdir(eli5_results_path)

    with open(eli5_results_path + "eli5_weights.html", "w") as file:
        file.write(
            eli5.formatters.html.format_as_html(eli5.explain_weights(algo, vec=vec, feature_names=names, top=20)))

    df_universals = pd.read_csv('../data/Universal_Annotation_Results_Selection.csv').sort_values('FILENAME')
    df_piper_2022 = pd.read_csv('../data/MinNarrative_ReaderData_Final_Selection.csv').sort_values('FILENAME')

    # Get results of passage with highest narrative probability - Universals
    max_np_df_universals = df_universals[df_universals['Probability.Narrative'] ==
                                         df_universals['Probability.Narrative'].max()]
    eli5_passage_results(df_piper_2022, df_universals, max_np_df_universals, 'max_universals', eli5_results_path,
                         algo, vec, names)

    # Get results of passage with lowest narrative probability - Universals
    min_np_df_universals = df_universals[df_universals['Probability.Narrative'] ==
                                         df_universals['Probability.Narrative'].min()]
    eli5_passage_results(df_piper_2022, df_universals, min_np_df_universals, 'min_universals', eli5_results_path,
                         algo, vec, names)

    # Get results of passage with highest narrative probability - Piper (2022)
    max_np_df_piper_2022 = df_piper_2022[df_piper_2022['Probability.Narrative'] ==
                                         df_piper_2022['Probability.Narrative'].max()]
    eli5_passage_results(df_piper_2022, df_universals, max_np_df_piper_2022, 'max_piper_2022', eli5_results_path,
                         algo, vec, names)

    # Get results of passage with lowest narrative probability - Piper (2022)
    min_np_df_piper_2022 = df_piper_2022[df_piper_2022['Probability.Narrative'] ==
                                         df_piper_2022['Probability.Narrative'].min()]
    eli5_passage_results(df_piper_2022, df_universals, min_np_df_piper_2022, 'min_piper_2022', eli5_results_path,
                         algo, vec, names)

    # Get results of passage with biggest difference - High Piper (2022), low Universals
    df_diff = pd.concat([df_universals['FILENAME'],
                         df_piper_2022['Probability.Narrative'] - df_universals['Probability.Narrative']],
                        axis=1).sort_values('Probability.Narrative')
    max_diff = df_piper_2022[df_diff['Probability.Narrative'] == df_diff['Probability.Narrative'].max()]
    eli5_passage_results(df_piper_2022, df_universals, max_diff, 'max_dif_hp_lu', eli5_results_path,
                         algo, vec, names)

    # Get results of passage with biggest difference - Low Piper (2022), high Universals
    min_diff = df_piper_2022[df_diff['Probability.Narrative'] == df_diff['Probability.Narrative'].min()]
    eli5_passage_results(df_piper_2022, df_universals, min_diff, 'max_dif_lp_hu', eli5_results_path,
                         algo, vec, names)


if __name__ == '__main__':
    model = 'word1'
    main()
