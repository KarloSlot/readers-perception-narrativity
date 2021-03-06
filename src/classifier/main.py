"""
- Run multiple experiments for different feature-categories.
- K-fold cross validated hyperparamter tuning for Logistic Regression
- Report the best f1-score along with precision, recall, AUROC, AUPRC, Weighted f1, and accuracy.

- Runs on main training dataset (~13k) with/without misclassifications.
- Runs on 2-class Reader-Annotated dataset. See main_3class for 3-class Reader-Annotated dataset.
"""

import vectorizer
import data_loader
import tuning
from collections import Counter

import os
import random
import numpy as np

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def run_experiments(algo, X, Y, name):
    """
    Run experiments for different feature categories as specified by "name".
    """
    if name == 'pos1':
        funct = vectorizer.pos_unigrams
    elif name == 'pos2':
        funct = vectorizer.pos_bigrams
    elif name == 'pos3':
        funct = vectorizer.pos_trigrams
    elif name == 'pos23':
        funct = vectorizer.pos_bitri_grams

    elif name == 'word1':
        funct = vectorizer.word_unigrams
    elif name == 'word2':
        funct = vectorizer.word_bigrams
    elif name == 'word3':
        funct = vectorizer.word_trigrams
    elif name == 'word23':
        funct = vectorizer.word_bitri_grams

    elif name == 'dep1':
        funct = vectorizer.dep_unigrams
    elif name == 'dep2':
        funct = vectorizer.dep_bigrams
    elif name == 'dep3':
        funct = vectorizer.dep_trigrams
    elif name == 'dep23':
        funct = vectorizer.dep_bitri_grams

    f1, auc, weighted_f1, prec, rec, accuracy, auprc, params = tuning.hyperparameter_tuning(algo, X, Y, funct,
                                                                                            NUMBER_OF_FOLDS,
                                                                                            three_class)
    print("F1:", f1)
    results_file.write(
        str(f1) + '\t' + str(auc) + '\t' + str(weighted_f1) + '\t' + str(prec) + '\t' + str(rec) + '\t' + str(
            accuracy) + '\t' + str(auprc) + '\t' + str(params) + '\n')


def main(three_class=False):
    """
    Run experiments for different feature-categories and different data-subsets.
    """
    kind = '5S'
    for feature_name in features:
        print("\n\n-----------------------\nRUNNING FOR: Kind =", kind, "| Feature =", feature_name)
        X, Y = data_loader.load_annotated_data(threshold=2.5)
        print("\nX: {} | Y: {} | Distribution: {} | Y preview: {}".format(len(X), len(Y), Counter(Y), Y[:3]))
        results_file.write(kind + '_' + str(len(X)) + '\t' + feature_name + '\t')
        run_experiments(algo_name, X, Y, feature_name)


if __name__ == '__main__':
    algo_name = 'rf'  # 'rf' or 'logreg' or 'svm'
    NUMBER_OF_FOLDS = 5

    three_class = False

    results_path = '../../results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    results_model_path = '../../results/model_results/'
    if not os.path.exists(results_model_path):
        os.mkdir(results_model_path)

    print("Running {}-fold CV | Algo = {} | Max-features = {}".format(NUMBER_OF_FOLDS, algo_name,
                                                                      vectorizer.MAX_FEATURES))
    results_path = results_model_path + algo_name + '__' + str(NUMBER_OF_FOLDS) + '_foldcv.txt'  # name of output file
    print("\n-------\nResults path:", results_path, "\n\n")
    results_file = open(results_path, "w")
    results_file.write("Data\tFeature\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\tParameters\n")
    features = ['pos1', 'pos2', 'pos3', 'pos23', 'word1', 'word2', 'word3', 'word23', 'dep1', 'dep2', 'dep3', 'dep23']

    main()
