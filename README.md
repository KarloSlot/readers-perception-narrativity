# READ ME (readers-perception-narrativity)
In this repository, all code can be found used in the Bachelor Information Science Thesis 'The Element of Surprise, Suspense and Curiosity: Annotation of Readers’ Perception to Detect Narrativity' by K.H.R. Slot. Parts of the code employed was provided by Piper et al. (2021), while the data was provided by Piper et al. (2022). For that, I am grateful.

DISCLAIMER: Due to legal reasons, I am not able to publish all data, i.e. MinNarrative, gathered by Piper et al. (2022). The selection of data used within this research, enriched with new data can be found in this repository, but all 17K+ files should be provided by Piper et al. (2022).


# Structure
## 'data'
In the subdirectory 'data', the data used in this research can be found. In the 'annotation_rounds', the results of individual annotation rounds can be found. Here are also the construction of the final data set be found. Crucial files for running the classifier and prediction files include:

**MinNarrative_ReaderData_Final.csv** - Small data set of corpus from Piper et al. (2022).

**MinNarrative_ReaderData_Final_Selection.csv** - Selection of 325 passages from the data set constructed by Piper et al. (2022). This includes annotation results and NLP features. This file is used for comparison with the results of our annotation and computational experiments.

**Universal_Annotation_Results_Selection.csv** - Selection of 325 passages from the data set constructed by Piper et al. (2022), but with unique annotations to reflect suspense, curiosity and surprise. The file includes NLP features as well. This file is used for comparison with the results of our annotation and computational experiments and train our computational models.

**keep_dataset_only.py** - Uses 'MinNarrative_ReaderData_Final.csv' to remove all MinNarrative Files from Piper et al. (2022) which are not needed to train the computational models. This files should be run before 'process_passages_booknlp.ipynb'.

**process_passages_booknlp.ipynb** - This notebook processes all available MinNarrative files from Piper et al. (2022). This produces a folder named 'BookNLP' which is used to train the computational models and should be put in this folder.

NOTE: All preprocessing and data gathering necessary to run the computational models has been done already. Therefore, to run the classifier, running files stored in 'data' is NOT needed.


# 'src'
This subdirectory is used to run the actual computational models, the files in here are sometimes part of a sub-subdirectory 'classifier'. This is specified when needed.

**classifier/config.py** - File used by Piper et al. (2021) to determine several NLP features of texts. This is not further used within this research, but since it is imported by Piper et al. (2021), it is worth putting in this repository.

**classifier/data_loader.py** - 

**predict.py** - 
