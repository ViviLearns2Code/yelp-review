from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from util import *
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import time

parquetpath = './dataset/final/'
trigram = Phraser.load("./vocab/trigram")
bigram = Phraser.load("./vocab/bigram")
dct = Dictionary.load("./gensim_dct")
reviews = pd.read_parquet(path=parquetpath)
reviews = reviews[reviews["length"]>5].sample(10000)

rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(reviews[["text"]],reviews["sentiment"])
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)
X_resampled.columns = reviews[["text"]].columns
y_resampled.columns = reviews[["sentiment"]].columns

model = TfidfModel(dictionary=dct)
t0 = time.time()
X_csc = apply_tfidf(dct,model,X_resampled,bigram,trigram)
t1 = time.time()
print("Applied tfidf:", t1-t0)
# use SVD only
reducer = TruncatedSVD(n_components=2)
X_reduced = reducer.fit_transform(X_csc)
points_svd=X_reduced
# use SVD for dimensionality reduction and then TSNE
reducer = TruncatedSVD(n_components=50)
X_reduced = reducer.fit_transform(X_csc)
t2 = time.time()
print("Applied SVD:", t2-t1)
tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=2500)
points_tsne = tsne.fit_transform(X_reduced)
t3 = time.time()
print("Applied TSNE:", t3-t2)
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
colormap={-1:"r",0:"y",1:"g"}
colors=list(map(lambda x: colormap[x], y_resampled["sentiment"].tolist()))
print("Number of samples:",y_resampled.shape[0])
ax1.scatter(points_svd[:,0],points_svd[:,1],s=5,c=colors)
ax2.scatter(points_tsne[:,0],points_tsne[:,1],s=5,c=colors)
plt.show()