import re
import time
from langid.langid import LanguageIdentifier, model
import pandas as pd
import numpy as np
import dask as da
import dask.dataframe as dd

'''
Runtime: ~3.5 hours
(total: 12101.699016094208 compute: 12100.365365028381)
'''

srcpath = './dataset/yelp_academic_dataset_review.json'
tgtpath = './dataset/processed/'

class Preprocessor():
    def __init__(self):
        pass

    def clean_text(self, text, identifier):
        '''
        Remove emails, urls
        Remove new lines, brackets, slashes, hyphens, duplicate spaces
        Add space between words with punctuation in between, e.g. "hello!?world
        '''
        missing_space = r"[A-Za-z][\.!?,\*]+[A-Za-z]"
        mail_url = r"([a-zA-Z0-9-_.]+@([a-zA-Z0-9-_]+\.)+[a-zA-Z]+)|([Hh][Tt]{2}[Pp][Ss]{0,1}:\/\/[a-zA-Z0-9-_.]+[a-zA-Z])|([Ww]{3}([\.[a-zA-Z0-9-_])+[a-zA-Z])"
        cont_space = r"\s{2,}"
        bracket_slash = r"[\)\(\\\/]"

        def _add_space(matchobj):
            expr = matchobj.group(0)
            return expr[0]+". "+expr[-1]

        if identifier.classify(text)[0] != "en":
            return "NON_ENGLISH_REVIEW"

        copy = text.replace("\n"," ")  
        copy = copy.replace("="," is ")  
        copy = re.sub(pattern=mail_url, repl="", string=copy)
        copy = copy.replace("-"," ")   
        copy = re.sub(pattern=bracket_slash,repl=" ", string=copy)              
        copy = re.sub(pattern=missing_space, repl=_add_space, string=copy)
        copy = re.sub(pattern=cont_space, repl=" ", string=copy)
        return copy
    
    def map_rating(self, rating):
        if rating <= 2:
            return -1
        if rating <= 3.5:
            return 0
        else:
            return 1

def transform_df(df):
    ''' dataframe '''
    print("transform_df called for partition...")
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)  #takes 2 seconds
    out = pd.DataFrame(columns=["text","sentiment"])
    preprocessor = Preprocessor()
    out["text"] = df["text"].apply(func=preprocessor.clean_text,args=(identifier,))
    out["sentiment"] = df["stars"].map(preprocessor.map_rating)
    return out

meta = pd.DataFrame(columns=["text","sentiment"])
meta["sentiment"] = meta["sentiment"].astype(np.int8)
meta["text"] = meta["text"].astype(str)

records = dd.read_json(srcpath, orient="records", blocksize=2**25, sample=5, encoding="utf-8")[["stars","text"]]
t0=time.time()
result = records.map_partitions(func=transform_df, meta=meta)
result_en = result[result["text"]!="NON_ENGLISH_REVIEW"].dropna()
parquet = dd.to_parquet(df=result_en, path=tgtpath, compute=False) 
t1=time.time()
parquet.compute(scheduler="processes")
t2=time.time()
print("total:",t2-t0, "compute:", t2-t1)
