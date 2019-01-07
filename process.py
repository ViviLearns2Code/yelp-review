import spacy
from spacy.tokens import Token
from spacy.symbols import ORTH, LEMMA, NORM

import dask as da
import dask.dataframe as dd

import pandas as pd
import numpy as np

from collections import Counter
import pickle
import uuid
import time

srcpath = './dataset/processed/'
tgtpath = './dataset/final/'

#--------- Configure spacy language model ------------
# 1. Add custom token attribute
Token.set_extension("cleaned", default="")
# 2. Load model
nlp = spacy.load('en_core_web_md', disable=["parser", "ner"])
# 3. Define custom stop words 
STOP_WORDS = ["be", "the", "and", "i", "a", "to", "it", "of", 
"have", "for", "in", "my", "that", "we", "they", "with", 
"you", "this", "do", "us", "them", "me", "their", "his", 
"him", "her"]
# 4. Define custom pipelines
def _filter(doc):
    ''' remove non-alphabetic tokens, punctuation '''
    exc_list  = ["n't","'m","'re"]
    return [token for token in doc if not token.is_space and not (nlp.vocab[token.lemma_.lower()].is_stop or nlp.vocab[token.lower_].is_stop) and not token.is_punct and (token.text in exc_list or token.is_alpha)]

def _lemmatize(doc):
    for token in doc:
        if token.lemma_ != "-PRON-":
            token._.cleaned = token.lemma_.lower()
        else: 
            token._.cleaned = token.lower_
    return doc

def _unknown(doc):
    for token in doc:
        if token.is_oov == True and token.text != "number": #spacy bug
            token._.cleaned = "UNK"
    return doc

def _finalize(doc):
    ''' drop unknown tokens and custom stop words '''
    return [token for token in doc if token._.cleaned != "" and token._.cleaned not in STOP_WORDS and token._.cleaned != "UNK"]

nlp.add_pipe(_filter, name="filter", after="tagger")
nlp.add_pipe(_lemmatize, name="lemmatize", after="filter")
nlp.add_pipe(_unknown, name="unknown", after="lemmatize")
nlp.add_pipe(_finalize, name="finalize", after="unknown")
# 5. Expand custom contractions
exc = [{ORTH: "whad", LEMMA: "what", NORM: "what"},
    {ORTH: "d'", LEMMA: "do", NORM: "do"},
    {ORTH: "ya", LEMMA: "you", NORM: "you"}]
nlp.tokenizer.add_special_case("whadd'ya",exc)
exc = [{ORTH: "whad", LEMMA: "what", NORM: "what"},
    {ORTH: "d", LEMMA: "do", NORM: "do"},
    {ORTH: "ya", LEMMA: "you", NORM: "you"}]
nlp.tokenizer.add_special_case("whaddya",exc)

#--------- Process dataframe ------------
def analyze_doc(doc, sentiment, adj_pos, adj_neu, adj_neg):
    ''' Collect adjectives per sentiment category '''
    for token in doc:
        if (token.pos_ == "ADJ" and token.tag_ == "JJ"):
            # exclude possessive adjectives
            if sentiment == 1:
                adj_pos[token._.cleaned] += 1
            elif sentiment == 0:
                adj_neu[token._.cleaned] += 1
            elif sentiment == -1:
                adj_neg[token._.cleaned] += 1


def analyze_df(df):
    ''' Process dataframe partition and write statistics to files '''
    print("Analyze dataframe...")
    out = pd.DataFrame(columns=["text","sentiment","length"])   
    adj_pos = Counter()
    adj_neg = Counter()
    adj_neu = Counter()
    length = []
    texts = []
    seq = [(doc,sentiment) for doc, sentiment in zip(df["text"],df["sentiment"])] 
    for doc, context in nlp.pipe(seq, as_tuples=True, n_threads=-1, batch_size=400):
        length.append(len(doc))
        analyze_doc(doc,context,adj_pos,adj_neu,adj_neg)
        texts.append(" ".join([token._.cleaned for token in doc]))
    
    f_adj_neg = 'adj_neg/'+uuid.uuid4().hex
    f_adj_pos = 'adj_pos/'+uuid.uuid4().hex
    f_adj_neu = 'adj_neu/'+uuid.uuid4().hex
    with open(f_adj_pos, 'wb') as file:
        pickle.dump(adj_pos, file, pickle.HIGHEST_PROTOCOL)
    with open(f_adj_neu, 'wb') as file:
        pickle.dump(adj_neu, file, pickle.HIGHEST_PROTOCOL)
    with open(f_adj_neg, 'wb') as file:
        pickle.dump(adj_neg, file, pickle.HIGHEST_PROTOCOL)
    
    out["text"] = texts
    out["sentiment"] = df["sentiment"]
    out["length"] = length
    return out

meta = pd.DataFrame(columns=["text","sentiment","length"])
meta["sentiment"] = meta["sentiment"].astype(np.int8)
meta["text"] = meta["text"].astype(str)
meta["length"] = meta["length"].astype(np.int64)

records = dd.read_parquet(path=srcpath)
t0 = time.time()
result = records.map_partitions(analyze_df, meta=meta)
parquet = dd.to_parquet(df=result[result["length"]>0], path=tgtpath, compute=False) 
parquet.compute(scheduler="processes")
t1 = time.time()
print(t1-t0)