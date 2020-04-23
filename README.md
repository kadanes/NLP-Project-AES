# NLP-Project-AES
Course project for CS 7650 NLP (Georiga Tech)


The repository has been structured into 3 main folders:
1. aes_ml : Machine Learning based approaches to AES (Parth)
2. aes_dl : Deep Learning based approaches to AES (Gaurav)
3. gender_age_models: Bias analysis in grading (Naila)


### AES using Machine Learning (aes_ml )

This folder has 2 notebooks that have the results. `aes_ml_exploration.ipynb` is the notebook that has all the exploration we did. It includes hyperparameter tuning along with the results. `aes_ml_results.ipynb` has the final results that were presented in the paper. Hence to get the final results from the paper please **run all cells in `aes_ml_results.ipynb`**

Besides these notebooks the other code for data preprocessing, feature extraction, data frame generation and training has been split in to various python files.
1. `feature_extraction.py`: Contains code to process essays and wordlists to generate various style and content based features.
- `sentence_to_word_list(sentence, ignore_stopwords, set_no)` is used to generate list of words and calculate word level style based features. E.g.: Number of stop words, number of words with greater than 5 characters, etc.
- `essay_to_sentences(essay, set_no, ignore_stopwords=False)` is used to convert an essay to list of sentences and pass those to `sentence_to_word_list`. It is also used to generated sentence level style based features, E.g.: number of sentences. This function also sums the word level features that `sentence_to_word_list` returns. 
- `get_avg_word_vec(words, model, wv_size)` is used to return the average word vector representation of a list of words. 
- `get_prompt_word_vecs(prompt, model, wv_size)` is used to get the average word vector representation of paragraphs from the story for type 2 prompts. 
- `get_word_count_vector(words, set_no)` is used to get a sparse word count vector representation using `word_to_index` maps.
- `get_prompt_vectors(prompt, set_no)` is used to get sparse word count vectors for paragraphs from the story for type 2 prompts.

2. `frame_maker.py`: Contains the code to generate pandas dataframes that will be used in training the models.
- 'makeDataFrame(data)' makes the features dataframe from the dataframe that contains essay data loaded from ASAP's csv.
- `split_in_sets(data, base_min_scores, base_max_scores)` is used split the features dataframe by set number of the essays and display other information about each sets like prompt type.

3. 
