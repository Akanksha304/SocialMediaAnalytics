# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:22:23 2017

@author: Akanksha
"""

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
 
    with open('C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/Akanksha_twitter_credentials.json', 'r') as f:
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
path = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/JSON FILES'
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
d = 'C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/JSON FILES'
text = open('C:/Users/ShrivastavaAkanksha/Documents/MIS_FALL 2017/DataScience/test.txt').read()
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