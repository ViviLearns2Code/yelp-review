import pickle
import os
from collections import Counter

from gensim.models.phrases import Phrases, Phraser
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import gensim, scipy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_wordcloud():
    pos = Counter()
    neu = Counter()
    neg = Counter()
    for filename in os.listdir('adj_pos/'):
        with open('adj_pos/'+filename, 'rb') as file:
            pos = pos + pickle.load(file)
    for filename in os.listdir('adj_neu/'):
        with open('adj_neu/'+filename, 'rb') as file:
            neu = neu + pickle.load(file)
    for filename in os.listdir('adj_neg/'):
        with open('adj_neg/'+filename, 'rb') as file:
            neg = neg + pickle.load(file)
    wc_pos = WordCloud(background_color="white", max_words=50)
    wc_pos.generate_from_frequencies(pos)
    wc_neu = WordCloud(background_color="white", max_words=50)
    wc_neu.generate_from_frequencies(neu)
    wc_neg = WordCloud(background_color="white", max_words=50)
    wc_neg.generate_from_frequencies(neg)
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(wc_pos, interpolation="bilinear")
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(wc_neu, interpolation="bilinear")
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(wc_neg, interpolation="bilinear")
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.show()


def get_stats(data: pd.DataFrame):
    size_pos = data[data["sentiment"]==1].count()
    size_neg = data[data["sentiment"]==-1].count()
    size_neu = data[data["sentiment"]==0].count()
    mean_pos = data[data["sentiment"]==1]["length"].mean()
    mean_neg = data[data["sentiment"]==-1]["length"].mean()
    mean_neu = data[data["sentiment"]==0]["length"].mean()
    max_pos = data[data["sentiment"]==1]["length"].max()
    max_neg = data[data["sentiment"]==-1]["length"].max()
    max_neu = data[data["sentiment"]==0]["length"].max()
    print("Total count:",data.count(),"Positive count:",size_pos,"Negative count:",size_neg,"Neutral count:",size_neu)
    print("Positive mean:",mean_pos,"Negative mean:",mean_neg,"Neutral mean:",mean_neu)
    print("Positive max:",max_pos,"Negative max:",max_neg,"Neutral max:",max_neu)

def plot_dist(data: pd.DataFrame):
    data_pos = data[data["sentiment"]==1]
    data_neg = data[data["sentiment"]==-1]
    data_neu = data[data["sentiment"]==0]
    plt.figure("Number of tokens")
    ax = sns.distplot(data_pos["length"], label="Positive", hist=False)
    sns.distplot(data_neu["length"], label="Neutral", hist=False)
    sns.distplot(data_neg["length"], label="Negative", hist=False)
    ax.set_xlabel('Number of tokens')
    plt.show()

def extract_phrases(df: pd.DataFrame):
    """
    Train bigram and trigram phrasers
    Input:
    - df: dataframe with column "text"
    """
    def wrapper(generator):
        for item in generator:
            yield item.text.split(" ")  

    vocab = Counter()
    vocab_final = Counter()
    bigram_phrases = Phrases(wrapper(df.itertuples()), min_count=5, threshold=1)
    bigram = Phraser(bigram_phrases)
    trigram_phrases = Phrases(bigram[wrapper(df.itertuples())], min_count=5, threshold=1)
    trigram = Phraser(trigram_phrases)
    bigram.save("./vocab/bigram")
    trigram.save("./vocab/trigram")

def create_dct(df: pd.DataFrame, bigram: gensim.models.phrases.Phraser,trigram: gensim.models.phrases.Phraser, save: bool=False):
    """
    Create dictionary from dataframe
    Input:
    - df: dataframe with column "text"
    - bigram: bigram phraser
    - trigram: trigram phraser
    - save: if true, vocabulary is saved in files
    """
    def wrapper_phrase(generator):
        for item in generator:
            ngram  = trigram[bigram[item.text.split(" ")]]
            yield ngram
    dct = Dictionary.from_documents(wrapper_phrase(df.itertuples()))
    dct.filter_extremes(no_below=1000, no_above=0.80, keep_n=150000)
    if save == True:
        dct.save_as_text("./gensim_dct.txt")
        dct.save("./gensim_dct")

def apply_tfidf(dct: gensim.corpora.Dictionary, model: gensim.models.TfidfModel, df: pd.DataFrame, bigram: gensim.models.phrases.Phraser, trigram: gensim.models.phrases.Phraser) -> scipy.sparse.csc_matrix:
    """
    Apply TF-IDF transformation
    Input:
    - dct: dictionary object
    - model: tfidf model
    - df: dataframe with column "text"
    - bigram: bigram phraser
    - trigram: trigram phraser
    """
    def wrapper_tfidf(generator):
        for item in generator:
            yield dct.doc2bow(trigram[bigram[item.text.split(" ")]])
    transformed_corpus = model[wrapper_tfidf(df.itertuples())]
    csc = corpus2csc(transformed_corpus,num_terms=len(dct)).transpose()    
    return csc