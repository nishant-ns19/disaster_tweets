import pandas as pd
import re
import fasttext

from textblob import TextBlob
import string
import preprocessor as p

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.random import RandomState
from nltk.corpus import stopwords
import csv
import sys
import nltk


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

print('Splitting data into 90% training data and 10% test data..')
print('----------------------TRAINING MODEL------------------------------')
model = fasttext.train_supervised('train.txt', lr=0.1, epoch=10, wordNgrams=2)
print('----------------------Applying classifier on test data------------------------------')
print('Calculating precision, re-call')
test_csv = pd.read_csv("test.csv")
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for index, row in test_csv.iterrows():
	processed_tweet = clean_tweets(p.clean(row['text']))
	[x,y] = model.predict(processed_tweet)
	prediction = int(x[0][-1])
	label = int(row['target'])
	true_pos += int(label == 0 and prediction == 0)
	true_neg += int(label == 1 and prediction == 1)
	false_pos += int(label == 1 and prediction == 0)
	false_neg += int(label == 0 and prediction == 1)
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
Fscore = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", Fscore)
print("Accuracy: ", accuracy)
model.save_model("model.bin")
print('Model Saved')



