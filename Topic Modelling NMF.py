# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:27:24 2017

@author: Akanksha
"""

#NMF and LDA Topic modeling for Texas

#Topic modeling using NMF
import os
import sys
import json
import re
import string
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.corpus import stopwords
from nltk import word_tokenize
from __future__ import division
from gensim import corpora, models, similarities, matutils
import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer





from nltk.stem.snowball import SnowballStemmer
ss = SnowballStemmer("english")

# Variables used 
import re
tweets = []
jsonFiles = []
poltrump=[]
subtrump=[]
trumpcorpus=[]
trump = ''
words2=[]


def remove_non_ascii(string):
    return ''.join(i if ord(i) <127 else ' ' for i in string)



cachedStopWords = set(stopwords.words("english"))
#add custom words
morewords= 'trump','make','RT','amp','rt','wants','cut','say','presid','fal','said','like','head','gt','look',
'dogg','peopl','yeah','agr', 'cut','artfund'
cachedStopWords.update(morewords)


#Merging json files
path = 'C:/Users/Akanksha304/DeltaTweets/Project1DS/Texas tweets'


for files in os.listdir(path):
    jsonFiles.append(files)
    
os.chdir(path)


for fname in jsonFiles:
    infile = open(fname).read()
    content = json.loads(infile)

    
    for i in range(len(content)):
        trumptweet = remove_non_ascii(content[i]['text']).encode('utf-8')
        trumptweet = re.sub(r"http\S+|@\S+"," ",trumptweet)
        trumptweet = re.sub(r"\d"," ",trumptweet)
        trumptweet = re.sub(r'[\']'," ",trumptweet)
        trumptweet = re.sub(r'[^A-Za-z0-9]+'," ",trumptweet)
        
      
        token=trumptweet.split()
        
        
        trumptweet = ' '.join([word for word in token if word.lower() not in cachedStopWords])
        trump += trumptweet + '\n '
        trumpcorpus.append(trump)

    trumpcorpus = [" ".join([ss.stem(r) for r in sentence.split(" ")]) for sentence in trumpcorpus]
    #print trumpcorpus
    
    
#with open('AllTexastweets.json' , 'w') as f1:
    #f1.write(trump)
    
    




# In[ ]:

import numpy as np  # a conventional alias
import glob
import os
import string
import nltk
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from sklearn import decomposition

vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    doc_term_matrix=vectorizer.fit_transform(trumpcorpus)    #req list as an input arg
print doc_term_matrix.shape
vocab = vectorizer.get_feature_names()

#Printing number of documents and number of unique words

print 'num of documents, num of unique words'
print doc_term_matrix.shape

num_topics = 20

clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(doc_term_matrix)

topic_words = []
num_topics = 20
num_top_words = 10

print vocab[100]

for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

print doc_term_matrix.shape   
for t in range(len(topic_words)):
    print ("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))
    
