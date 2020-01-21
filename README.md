# Project Overview

This project pairs the semester-long efforts of a Capstone team at NYU (the authors of this repository), with previous work accomplished in the field of biased language detection by FairFrame, a company striving to bring “enhanced thinking about diversity and inclusion” to professional settings. They provide a technology platform which allows users to upload employee evaluations and receive feedback on words they are using which may contain some form of gender bias. However, their current algorithm employs a direct text-matching algorithm of the review against a list of words identified as biased, without consideration of various contexts in which the word may be used in a non-biased context (e.g., “She is difficult to work with” vs “She excels at difficult tasks”).

This repository contains code serving as a proof-of-concept for future implementation by the company on the task of determining whether a list of flagged words are being used in a problematic or non-problematic context.

## Dataset 

The dataset contains proprietary information gathered by the company, and thus are hidden. The dataset we built our model with contains the following:
- A list of flagged words which may connote gender bias
- A set of 110K reviews, scraped from various websites wherein companies may provide feedback for freelancing employees
- Of the above 110K reviews, the NYU & FairFrame teams worked together to annotate a subset of 5K reviews, based on whether the flagged word was being used in a problematic manner

## Modeling

Our modelling efforts utilized a semi-supervised learning setting to make use of both the labelled & unlabelled dataset described above. Our general approach was to train embedding representations of each sentence containing a flagged word, and then cluster into two distinct groups representing whether the sentence does or does not contain some form of gender bias. We utilize the F1-score metric as well as UMAP visualizations to assess model performance and separability of the clusters.

## UMAP Visualization

Each embedding from each model is saved from the code at the end of the model-training notebook.  For making plots using these saved models, use ‘plot_embedding.ipynb’ in UMAP folder. Please note that this notebook will save plots for each model once so every time you have to comment/uncomment model_name and model_folder.

## Repository Organization:

1) `data` folder -- contains scripts for initial processing of raw datasets: filtering reviews for only those containing flagged words, tokenizing, and converting to dataframe
- `master_df_labeled.xlsx` = contains labelled (& unlabelled) reviews
- dataloaders

2) `data_prep` folder -- notebooks to process data and generate dataloaders for models
- `get_all_reviews` and `filter_reviews` -- data processing notebooks specific to FairFrame’s proprietary dataset. The output is a DataFrame with tokenized review sentences, the flagged word, and the index of the flagged word
- `label_data_samples` -- updates a master data file with manual annotations of whether or not word was flagged in problematic context. This also contains code for creating data batches for manual annotation.
- `generate_dataloaders`, `generate_dataloaders.py`, and `bert_dataloaders` -- creates dataloaders required for model training

3) `model_training` folder -- contains scripts for baseline models, as well as LSTM and BERT
- `unsupervised_baseline_randomized_embeddings` -- trains unsupervised baseline model, with randomly-initialized embeddings
- `unsupervised_baseline_glove_embeddings` -- trains unsupervised baseline model, initialized with pretrained GloVe word embeddings
- `semisupervised_baseline_randomized_embeddings’ -- trains semi-supervised baseline model, with randomly-initialized embeddings
- `semisupervised_baseline_glove_embeddings` -- trains semi-supervised baseline model, initialized with pretrained GloVe word embeddings
- `lstm_model` - trains an LSTM model to generate contextualised embeddings for the reviews, initialized with pretrained GloVe word embeddings
- `bert_model` - finetunes the pretrained BERT base and BERT large models for our task

4) `model_output` folder -- contains subfolders for each model architecture. Within each subfolder are additional subfolders, saving the output for each hyperparameter setting trained.

5) `evaluation` folder -- contains scripts and notebooks for evaluating our model. Many models will import the `evaluation.py` script, which enables calculations of metrics such as accuracy, precision, recall, and F1-score over the validation set of labelled reviews

6) `umap` folder -- visualization code and saved images
- Contains subfolders for each model. Within each subfolder are pickle files containing trained embeddings, the labels and UMAP visualizations of the embeddings. Note that we only use best-performing hyperparameter configuration for each model type.
- `plot_embedding.ipynb` -- code to generate plots. Please note that this notebook will save plots for each model once so every time you have to comment/uncomment model_name and model_folder.