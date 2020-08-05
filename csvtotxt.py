import pandas as pd
import re
import os
from numpy.random import RandomState

from textblob import TextBlob
import string
import preprocessor as p

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.random import RandomState
from nltk.corpus import stopwords



emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

emoticons = emoticons_happy.union(emoticons_sad)

def clean_tweets(tweet):
    tweet = tweet.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = emoji_pattern.sub(r'', tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


df=pd.read_csv("data.csv")
cat = open('train.txt','w')
cat1 = open('test.txt','w')
rng = RandomState()
print('Split data into 90% training data and 10% test data..')
trainData = df.sample(frac=0.90, random_state=rng)
testData = df.loc[~df.index.isin(trainData.index)]
testData.to_csv('test.csv', index=False)

print('Creating Train Data..')
for index, row in trainData.iterrows():
    cat.write('__label__'+str(row['target'])+' '+clean_tweets(p.clean(row['text']))+'\n')

print('Creating Test Data..')
for index, row in testData.iterrows():
    cat1.write('__label__'+str(row['target'])+' '+clean_tweets(p.clean(row['text']))+'\n')



