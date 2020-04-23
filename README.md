# NLP-Project-AES
Course project for CS 7650 NLP (Georiga Tech)


The repository has been structured into 3 main folders:
1. aes_ml : Machine Learning based approaches to AES (Parth)
2. aes_dl : Deep Learning based approaches to AES (Gaurav)
3. gender_age_models: Bias analysis in grading (Naila)


### AES using Machine Learning (aes_ml)

This folder has 2 notebooks that have the results. `aes_ml_exploration.ipynb` is the notebook that has all the exploration we did. It includes hyperparameter tuning along with the results. `aes_ml_results.ipynb` has the final results that were presented in the paper. Hence to get the final results from the paper please **run all cells in `aes_ml_results.ipynb`**

Besides these notebooks the other code for data preprocessing, feature extraction, data frame generation and training has been split in to various python files.


**1. `feature_extraction.py`:** Contains code to process essays and wordlists to generate various style and content based features.
- `sentence_to_word_list(sentence, ignore_stopwords, set_no)` is used to generate list of words and calculate word level style based features. E.g.: Number of stop words, number of words with greater than 5 characters, etc.
- `essay_to_sentences(essay, set_no, ignore_stopwords=False)` is used to convert an essay to list of sentences and pass those to `sentence_to_word_list`. It is also used to generated sentence level style based features, E.g.: number of sentences. This function also sums the word level features that `sentence_to_word_list` returns. 
- `get_avg_word_vec(words, model, wv_size)` is used to return the average word vector representation of a list of words. 
- `get_prompt_word_vecs(prompt, model, wv_size)` is used to get the average word vector representation of paragraphs from the story for type 2 prompts. 
- `get_word_count_vector(words, set_no)` is used to get a sparse word count vector representation using `word_to_index` maps.
- `get_prompt_vectors(prompt, set_no)` is used to get sparse word count vectors for paragraphs from the story for type 2 prompts.


**2. `frame_maker.py`:** Contains the code to generate pandas dataframes that will be used in training the models.
- `makeDataFrame(data)` makes the features dataframe from the dataframe that contains essay data loaded from ASAP's csv.
- `split_in_sets(data, base_min_scores, base_max_scores)` is used split the features dataframe by set number of the essays and display other information about each sets like prompt type.


3. **`keys.py`:** Contains of all the keys that are used to access dataframe columns


4. **`prompts_reader.py`:** Contains the code to load paragraphs from story for type 2 prompts from the `./prompts` directory.


5. **`trainer.py`:** Contains the code to train ML models and evaluate the MQWK score accross 5 folds.
- `get_all_classifiers()` is used to get a list of clasifiers that was used in the exploration phase before Linear Regression was chosen.
- `create_word_vecs(X, wv_model, wv_size, essay_wordvecs)` is used to generate the average word vector representations for word from essays in a dataframe.
- `create_sim_from_word_vecs(X, data, wv_model, wv_size, essay_wordvecs)` is used to generate similarity between the essay's average word vector representation based with average word vector based representation of paragraphs from story of type 2 prompt.
- `evaluate(X, y, data=None, model = LinearRegression(), plot=False, wordvec=False, wv_size=300, min_count=30, context=10, sample=0, lsa=False, wordvec_sim=False)` is used to train the machine learning model across 5 fold cross validation and compute the value of MQWK score.

Besides these files, all the imports have been defined in python files that have been named as `requirements_*`. 

**1. `requirements_base.py`:** Contains all the basic library imports.

**2. `requirements_feature.py`:** Contains imports for functions defined in `feature_extraction.py`.

**3. `requirements_frame.py`:** Contains  imports for functions defined in `frame_maker.py`.

**4. `requirements_key.py`:** Contains imports for dictionarys and keys defined in `keys.py`.

**5. `requirements_trainer.py:`** Contains imports for functions defined in `trainer.py`.

