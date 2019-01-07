# Sentiment Analysis with Yelp Reviews

Source code for 
* https://vivilearns2code.github.io/nlp/2018/11/11/getting-started-with-nlp-part-i.html
* https://vivilearns2code.github.io/nlp/2018/11/18/getting-started-with-nlp-part-ii.html

## Requirements
* gensim 3.5.0
* scikit-learn 0.20.1
* wordcloud 1.5.0
* spacy 2.0.13
* imbalanced-learn 0.4.3
* dask 0.20.0

## Files
The script `preprocess.py` converts the original json file to a roughly processed parquet file. The next script, `process.py`, uses spaCy for further processing and text analysis. To vectorize the processed review texts, bigrams and trigrams need to be identified and based on that, a dictionary needs to be created. These and other methods are defined in `util.py`. Training happens in `training.py`, visualization in `visualization.py`.

The trigram phraser model was too big, which is why I left out the bigram and trigram phraser models. The dictionary can be loaded from `gensim_dct` and the trained model from `sentiment_classifier.sav`.