# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:22:25 2017

@author: Akanksha
"""

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
 
    
    with open('C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/Akanksha_twitter_credentials.json', 'r') as f:
        credentials = json.load(f)
             
#create your own app to get consumer key and secret

    CONSUMER_KEY = credentials['CONSUMER_KEY']
    CONSUMER_SECRET = credentials['CONSUMER_SECRET']
    ACCESS_TOKEN = credentials['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']
    
    try:
        stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

#for other states below mentioned coordinates were passed as an args
#Colorado (-108.734972,37.286814,-101.601546,41.090279)
#California(-122.286607,32.339765,-115.531159,41.595855)                       
#Florida(-83.711258,24.443950,-79.242430,31.273148)                        
#New york(-78.976756,40.176314,-71.160950,45.319524 
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

path3 = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/TexastoMisiTweets'
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
text = open('C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/TexastoMissitweets/Stemmedwords.txt').read()
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


