# READ ME (readers-perception-narrativity)
In this repository, all code can be found used in the Bachelor Information Science Thesis "The Element of Surprise, Suspense and Curiosity: Annotation of Readers’ Perception to Detect Narrativity" by K.H.R. Slot. Parts of the code employed was provided by Piper et al. (2021), while the data was provided by Piper et al. (2022). For that, I am grateful.

DISCLAIMER: Due to legal reasons, I am not able to publish all data, i.e. all MinNarrative files, gathered by Piper et al. (2022). The selection of data used within this research, enriched with new data can be found in this repository, but all 17K+ files should be provided by Piper et al. (2022).


# Structure
**Annotation Guidebook.pdf** - Guidelines used for annotating passages based on suspense, curiosity and surprise.

## 'data'
In the subdirectory 'data', the data used in this research can be found. In the 'annotation_rounds', the results of individual annotation rounds can be found and the functions to calculate IRR. Here are also the construction of the final data set be found. Crucial files for running the classifier and prediction files include:

**MinNarrative_ReaderData_Final.csv** - Small data set of corpus from Piper et al. (2022).

**MinNarrative_ReaderData_Final_Selection.csv** - Selection of 325 passages from the data set constructed by Piper et al. (2022). This includes annotation results and NLP features. This file is used for comparison with the results of our annotation and computational experiments.

**Universal_Annotation_Results_Selection.csv** - Selection of 325 passages from the data set constructed by Piper et al. (2022), but with unique annotations to reflect suspense, curiosity and surprise. The file includes NLP features as well. This file is used for comparison with the results of our annotation and computational experiments and train our computational models.

**keep_dataset_only.py** - Uses 'MinNarrative_ReaderData_Final.csv' to remove all MinNarrative Files from Piper et al. (2022) which are not needed to train the computational models. This files should be run before 'process_passages_booknlp.ipynb'.

**process_passages_booknlp.ipynb** - This notebook processes all available MinNarrative files from Piper et al. (2022). This produces a folder named 'BookNLP' which is used to train the computational models and should be put in this folder.

NOTE: All preprocessing and data gathering necessary to run the computational models has been done already in this research. If you wish to gather the BookNLP data, you should obtain the MinNarrative files from Piper et al. (2022), run keep_dataset_only.py, followed by process_passages_nlp.ipynb. The Python Notebook was constructed to work using Google Colab, ensure that the paths in the file are the same as those on your Google Driv. Make sure that the outcoming 'BookNLP' directory is unzipped in the folder 'data'. Please contact me for more information (For LinkedIn, see Profile).

## 'src'
This subdirectory is used to run the actual computational models, the files in here are sometimes part of a sub-subdirectory 'classifier'. This is specified when needed.

**classifier/data_loader.py + classifier/features.py + classifier/tuning.py + classifier/vectorizer.py** - Files used by Piper et al. (2021) to determine several NLP features of texts, construct and load training data and train computational models.

**classifier/main.py** - File to run when training computation models to detect narrativity, returns a file which shows evaluation metrics, determining the best model. These returned files can be found in results/model_results. Requires BookNLP.

**predict.py** - File to predict the narrative probabilities using computational methods (default model is 'word1', since this is the best performing model). This file also provide analysis of prominent passages, such as passages with a high narrative probability using ELI5. These results can be found in results/eli5_results. Requires BookNLP.

## 'results'
In this subdirectory are two other directories: (1) eli5_results and (2) model_results. In eli5_results, the produced ELI5 from predict.py can be found as per passage (accompanied with some passage information, such as narrative probability) and the general weights per feature, or word. In model_results, the results from main.py can be found, where the score of each pair of method and feature components can be found, e.g. RF with word-unigrams.

## 'analysis'
This subdirectory is used for storing graphs and plots construted by correlation.py to illustrate the correlation between aspects of our experiments and those of Piper et al. (2022). These plots can also be found in the main paper.

# References
Andrew Piper, Sunyam Bagga, Laura Monteiro, Andrew Yang, Marie Labrosse, and Yu Lu Liu. 2021. Detecting narrativity across long time scales. *Proceedings http://ceur-ws.org ISSN*, 1613:0073.

Andrew Piper, Sunyam Bagga, Laura Monteiro, Andrew Yang, Marie Labrosse, and Yu Lu Liu. 2022. Unpublished data.
