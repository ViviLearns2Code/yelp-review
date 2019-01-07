from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from imblearn.under_sampling import RandomUnderSampler

import time
from util import *

pd.set_option('display.max_colwidth', -1)
parquetpath = './dataset/final/'
trigram = Phraser.load("./trigram")
bigram = Phraser.load("./bigram")
dct = Dictionary.load("./gensim_dct")
reviews = pd.read_parquet(path=parquetpath)
reviews = reviews[reviews["length"]>5]
#X_train, X_test, y_train, y_test = train_test_split(reviews[["text"]],reviews["sentiment"],train_size=0.8)

## lower metrics compared to alternative with weighting only
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(reviews[["text"]],reviews["sentiment"])
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)
print("Resampled size:", y_resampled.shape)
X_resampled.columns = reviews[["text"]].columns
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,train_size=0.8)

model = TfidfModel(dictionary=dct)
t0 = time.time()
#~15 thousand features
X_train_csc = apply_tfidf(dct,model,X_train,bigram,trigram)
t1 = time.time()
print("Time for tfidf:",t1-t0)
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train_csc,y_train)
t2 = time.time()
print("Training:",t2-t1)
X_test_csc = apply_tfidf(dct,model,X_test,bigram,trigram)
res = lr.predict(X_test_csc)
# persist model
#joblib.dump(lr, "./sentiment_classifier.sav") 

print("Number positives in train:",np.sum(y_train==1),"Number negatives in train:",np.sum(y_train==-1),"Number neutrals in train:",np.sum(y_train==0))
print("Number positives in test:",np.sum(y_test==1),"Number negatives in test:",np.sum(y_test==-1),"Number neutrals in test:",np.sum(y_test==0))
pre_pos = precision_score(y_true=y_test.values, y_pred=res, average="weighted", labels=[1])
rec_pos = recall_score(y_true=y_test.values, y_pred=res, average="weighted", labels=[1])
pre_neg = precision_score(y_true=y_test.values, y_pred=res, average="weighted", labels=[-1])
rec_neg = recall_score(y_true=y_test.values, y_pred=res, average="weighted", labels=[-1])
pre_neu = precision_score(y_true=y_test.values, y_pred=res, average="weighted",labels=[0])
rec_neu = recall_score(y_true=y_test.values, y_pred=res, average="weighted", labels=[0])
pre = precision_score(y_true=y_test.values, y_pred=res, average="weighted")
rec = recall_score(y_true=y_test.values, y_pred=res, average="weighted")
acc = accuracy_score(y_true=y_test.values, y_pred=res)
print("Metrics for positive class")
print("Precision:",pre_pos,"Recall:",rec_pos)
print("Metrics for negative class")
print("Precision:",pre_neg,"Recall:",rec_neg)
print("Metrics for neutral class")
print("Precision:",pre_neu,"Recall:",rec_neu)
print("Aggregated metrics")
print("Precision:",pre,"Recall:",rec)