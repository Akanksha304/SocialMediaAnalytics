# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:00:35 2017

@author: Akanksha
"""


# coding: utf-8

# In[ ]:

# 10k Trump tweets collection in chunks of 500
from twython import TwythonStreamer
import sys
import os
import json
import time
from datetime import datetime
 
# Variables used 
tweets = []
jsonFiles = []
TrumpFiles = []

dt=str(datetime.now().time())[:8].translate(None,':')

    
class MyStreamer(TwythonStreamer):
    '''our own subclass of TwythonStreamer'''
 
    # overriding
    def on_success(self, data):
        
        if 'lang' in data and data['lang'] == 'en':
            tweets.append(data)
            print 'received tweet #', len(tweets), data['text'][:100]
           
            
        if len(tweets) >= 500:
            self.store_json()
            self.disconnect()
 
    # overriding
    def on_error(self, status_code, data):            
        print status_code, data
        self.disconnect()
        
    def store_json(self):
        with open('tweet_stream_{}_{}_{}.json'.format(dt,keyword, len(tweets)), 'w') as f:
            json.dump(tweets, f, indent=4)
            
if __name__ == '__main__':
 
    with open('C:/Users/Akanksha304/Desktop/Project/akanksha_Twitter_credentials.json','r') as f:
        
        credentials = json.load(f)
             
#create your own app to get consumer key and secret
    CONSUMER_KEY = credentials['CONSUMER_KEY']
    CONSUMER_SECRET = credentials['CONSUMER_SECRET']
    ACCESS_TOKEN = credentials['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']
 
    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
 
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    else:
        keyword = 'trump'
 
    stream.statuses.filter(track=keyword)
    


# In[ ]:

# Sentiment Analysis on 10k Trump tweets
import os
import sys
import json
from textblob import TextBlob
import matplotlib.pyplot as plot
import numpy as np

# Variables used 
tweets = []

jsonFiles = []
trump = ''
pola = []
subj = []

#Defining function to remove tweets with non-ascii characters
def remove_non_ascii(string):
    return ''.join(i if ord(i) <127 else ' ' for i in string)

#Merging json files
path = 'C:/Users/kudva/Desktop/UTA SEM 2/DS Project/JSON FILES'
path = 'C:/Users/AkankshaShrivasatca'
for files in os.listdir(path):
    jsonFiles.append(files)
    #print jsonFiles
os.chdir(path)

for lines in jsonFiles:
    infile = open(lines).read()
    content = json.loads(infile)
    
    for i in range(len(content)):
        trumptweet = remove_non_ascii(content[i]['text']).encode('utf-8')
        trump +=trumptweet + '\n'
        senseTrump = TextBlob(trumptweet)
        pola.append(senseTrump.sentiment.polarity)
        subj.append(senseTrump.sentiment.subjectivity)
        

#Code for plotting histogram with Polarity scores
plot.hist(pola, bins = 30)
plot.xlabel('Polarity Score')
plot.ylabel('Tweet Counts')
plot.grid(True)
plot.savefig('Polarity.pdf')
plot.show()

#Code for plotting histogram with Subjectivity Scores
plot.hist(subj, bins = 30)
plot.xlabel('Subjectivity Score')
plot.ylabel('Tweet Counts')
plot.grid(True)
plot.savefig('Subjectivity.pdf')
plot.show()

print('Average of Polarity Scores: {}'.format(np.mean(pola)))
print('Average of Subjectivity Scores: {}'.format(np.mean(subj)))

with open('AllTrumpTweets.json' , 'w') as f1:
    f1.write(trump)
    


# In[ ]:

# Wordcloud for 10k tweets
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
import numpy as np
from os import path

# appending words to stopwords as per judgement 
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('trump')
stopwords.append('donald')
stopwords.append('realdonaldtrump')
stopwords.append('one')
stopwords.append('know')
stopwords.append('need')
stopwords.append('presid')
stopwords.append('amp')
stopwords.append('indic')
stopwords.append('akhdr')
stopwords.append('lzhyaox')
stopwords.append('go')
stopwords.append('think')
stopwords.append('rt')
stopwords.append('https')
stopwords.append('co')
stopwords.append('say')
stopwords.append('us')
stopwords.append('gt')
stopwords.append('intel')

# Read the stemmed words text
d = 'C:/Users/Akanksha304/Desktop/Project/JSON FILES'
text = open('C:/Users/Akanksha304/Desktop/Project/JSON FILES/test.txt').read()
text2 = ''
for word3 in text.split():
    if len(word3)== 1 or word3 in stopwords:
        continue
    text2 += ' {}'.format(word3)

#Mask image in form of flag
image_mask = np.array(Image.open(path.join(d, "spectral.jpg")))

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=45, mask =image_mask).generate(text2) 

#Storing
wordcloud.to_file(path.join(d, "spectralWC.jpg"))

# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure()
#plt.imshow(wordcloud)
#plt.imshow(image_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:

#Topic Modeling - NMF and LDA Akanksha


# In[ ]:

#Collecting tweets based on location using latitude and longitude for Texas 

#Similarly tweets were collected for locations California, Colorado, Florida and New York

from twython import TwythonStreamer
import json
import time
from datetime import datetime
 
# Variables used 
tweets = []
jsonFiles = []
TrumpFiles = []
Caltweets = []

dt=str(datetime.now().time())[:8].translate(None,':')

    
class MyStreamer(TwythonStreamer):
    '''our own subclass of TwythonStremer'''
 
    # overriding
    def on_success(self, data):
        
        if 'lang' in data and data['lang'] == 'en' and 'trump' in data['text'].lower():
            Caltweets.append(data)
            print 'received tweet #', len(Caltweets), data['text'][:100]
           
            
        if len(Caltweets) >= 100:
            self.store_json()
            self.disconnect()
 
    # overriding
    #def on_error(self, status_code, data):            
        #print status_code, data
        #self.disconnect()
        
    def on_error_catch(self):
        print 'Saving tweets collected before error occurred'
        self.store_json()
        
    def store_json(self):
        with open('tweet_stream_{}_{}_{}.json'.format(dt, 'Texas', len(Caltweets)), 'w') as f2:
            json.dump(Caltweets, f2, indent=4)
            

if __name__ == '__main__':
 
    
    with open('C:/Users/ShrivastavaAkanksha/Documents/MIS_Spring 2017 Courses/Data Science/twitter_project_credentials.json', 'r') as f:
        credentials = json.load(f)
             
#create your own app to get consumer key and secret

    CONSUMER_KEY = credentials['CONSUMER_KEY']
    CONSUMER_SECRET = credentials['CONSUMER_SECRET']
    ACCESS_TOKEN = credentials['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']
    
    try:
        stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
 
   
# Coordinates combining location Texas to Mississipi
  
        stream.statuses.filter(locations = [-103.08, 28.93, -88.05, 36.25 ])
        
    except:
        stream.on_error_catch()


# In[ ]:

# Sentiment Analysis for Texas based tweets
import os
import json
import sys
import json
from textblob import TextBlob
import matplotlib.pyplot as plot
import numpy as np
import re 

Texastweets = []
#trumpTexas = []
Texasfile = ''
jsonFilesT = []
polaT = []
subjT = []
texascorpus = []

#Removing non-ascii characters from the tweets
def remove_non_ascii(string):
    return ''.join(i if ord(i) <127 else ' ' for i in string)

path3 = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_Spring 2017 Courses/Data Science/TexastoMissi'
for fnames in os.listdir(path3):
    jsonFilesT.append(fnames)
    #print jsonFilesT
    
os.chdir(path3)

for filenames in jsonFilesT:
    texfile = open(filenames).read()
    texcontent = json.loads(texfile)

for w in range(len(texcontent)):
    Texastweets = remove_non_ascii(texcontent[w]['text']).encode('utf-8')
    Texastweets = re.sub(r"http\S+|@\S+", " ", Texastweets)
    Texastweets = re.sub(r"\d", " ", Texastweets)
        #trumptweet = content[i]['text']
    Texasfile += Texastweets + '\n'
    #print Texasfile
    texascorpus.append(Texasfile) # dtm takes list of strings. Hence we created a list
    #print texascorpus
    senseTexas = TextBlob(Texastweets)
    polaT.append(senseTexas.sentiment.polarity)
    subjT.append(senseTexas.sentiment.subjectivity)

#Code for plotting histogram with Polarity scores

plot.hist(polaT, bins = 30)
plot.xlabel('Polarity Score for Texas')
plot.ylabel('Tweet Counts of Texas')
plot.grid(True)
plot.savefig('PolarityTexas.pdf')
plot.show()

#Code for plotting histogram with Subjectivity Scores
plot.hist(subjT, bins = 30)
plot.xlabel('Subjectivity Score for Texas')
plot.ylabel('Tweet Counts for Texas')
plot.grid(True)
plot.savefig('SubjectivityTexas.pdf')
plot.show()
       
# Average Polarity and Subjectivity scores for Texas 
print('Average of Polarity Scores: {}'.format(np.mean(polaT)))
print('Average of Subjectivity Scores: {}'.format(np.mean(subjT)))    

with open('AllTexastweets.json' , 'w') as f4:
    f4.write(Texasfile)  
    

 


# In[ ]:

# Wordcloud for Texas based tweets
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
import string

# appending words to stopwords as per judgement 
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('trump')
stopwords.append('donald')
stopwords.append('realli')
stopwords.append('let')
stopwords.append('say')
stopwords.append('noth')
stopwords.append('presid')
stopwords.append('pres')
stopwords.append('peopl')

#Removing punctuation and digits from Texas tweets

p = string.punctuation
d = string.digits

table_p = string.maketrans(p, len(p) * " ")
table_d = string.maketrans(d, len(d) * " ")
p1=Texasfile.translate(table_p)
p2=p1.translate(table_d)

newlist=p2.split()

words2 = []
for w in newlist:
    if w.lower() not in stopwords and len(w) > 1:
           words2.append(w)


#Stemming process
ss = SnowballStemmer("english")
ste=[]
for words1 in words2:
    q=ss.stem(words1)
    ste.append(q)


# Read the stemmed words text
text = open('C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/TexastoMissi tweets/Stemmedwords.txt').read()
text2 = ''
for word3 in text.split():
    if len(word3)== 1 or word3 in stopwords:
        continue
    text2 += ' {}'.format(word3)

# Generate a word cloud image

wordcloud = WordCloud(max_font_size=45).generate(text2) 

# Display the generated image
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



# In[ ]:

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
path = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/Texas tweets'
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
    




# In[ ]:

#Topic Modeling using LDA

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
from gensim import models
import gensim
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora


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
path = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/California tweets'
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
    
    
    
#with open('AllTrumpTweets.json' , 'w') as f1:
    #f1.write(trump)

    
texts = [[word for word in document.lower().split() if word not in cachedStopWords]
         for document in trumpcorpus]

dic = corpora.Dictionary(texts)

corpus = [dic.doc2bow(text) for text in texts]


NUM_TOPICS = 10
model = gensim.models.LdaModel(corpus, 
                                 num_topics=10, 
                                 id2word=dic, 
                                 update_every=1, 
                                 passes=20)


model.print_topics()

