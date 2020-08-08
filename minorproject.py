# -*- coding: utf-8 -*-
"""MinorProject.ipynb

Taking the positive and negative reviews of 2 politician: Politican X and Politician Y
"""


"""Working for politician X"""

with open('PoliticalLeaderX.txt', 'r') as f2:
    data = f2.read()
    #print(data)

import nltk
nltk.download('punkt')

sentences=nltk.sent_tokenize(data)

from  textblob import TextBlob
blob = TextBlob(data.strip())

import csv
header_name = ['labels', 'data']
with open('sentiment.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=header_name)
    writer.writeheader()

for sent in blob.sentences:
  #print(sent.sentiment.polarity)
  with open('sentiment.csv', 'a') as file:
    pol=sent.sentiment.polarity
    if(pol<0):
      pol=-1
    else:
      pol=1
    writer = csv.DictWriter(file, fieldnames=header_name)
    info = {
                'labels': pol,
                'data': sent
            }
    writer.writerow(info)

import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

import pandas as pd
df=pd.read_csv('sentiment.csv')

type(df['data'][8])

df.head

from nltk.tokenize import word_tokenize
allowed_word_types = ["J"]
all_words=[]
for p in  df['data']:
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    tokenized = word_tokenize(cleaned)
    
    stopped = [w for w in tokenized if not w in stopwords.words('english')]
    
    pos = nltk.pos_tag(stopped)
    
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

"""Bag of words model"""

# creating a frequency distribution of each adjectives.
all_words = nltk.FreqDist(all_words)

# listing the 1000 most frequent words
word_features = list(all_words.keys())[:1000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

import random
dataset=[]
i=0
for p in df['data']:
  feature=find_features(p)
  dataset.append([feature,df['labels'][i]])
  i+=1
random.shuffle(dataset)

print(len(dataset))# dataset is a bag of model of vectors

training_set = dataset[:10000]
testing_set = dataset[10000:]

"""Machine Learning Classifiers"""

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# training various models by passing in the sklearn models into the SklearnClassifier from NLTK 

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)

from sklearn.metrics import f1_score, accuracy_score
ground_truth = [r[1] for r in testing_set]
predictions = []
f1_scores = {}
Li={0:MNB_clf,1:BNB_clf,2:LogReg_clf,3:SVC_clf}
print(Li[0])
for x in Li:
  predictions=[]
  for r in testing_set:
    predictions.append(Li[x].classify(r[0]))
  if(x==0):
    str='Multinomial Naive Baeyes'
  elif(x==1):
    str='Bernoulli Naive Baeyes'
  elif(x==2):
    str='Logistic Regression'
  else:
    str='Support Vector'
  print(str,'classifier accuracy :',accuracy_score(ground_truth,predictions))
  print(str,'f1 score :',f1_score(ground_truth,predictions))

"""So we clearly see that Logistic regression classifier and Support vector perfectly classifies our dataset
But since SVM has more f1-score so we will make predictions based on SVM
"""

predictions=[]
for r in testing_set:
  predictions.append(SVC_clf.classify(r[0]))
print(predictions)

positive=0
negative=0
for i in range(0,len(predictions)):
  if predictions[i]==1:
    positive=positive+1
  else:
    negative=negative+1

print(positive,negative)

"""Sentimental Analysis"""

positive_percent=positive/(positive+negative)
negative_percent=negative/(positive+negative)

print(positive_percent*100,negative_percent*100)

"""So nearly 92% people are in favour of politician X and around 8% people do not favour him

Taking new Data for another politician (politician Y)
"""

data=''
with open('PoliticalLeaderY.txt', 'r') as f2:
    data = f2.read()
    #print(data)

from  textblob import TextBlob
blob = TextBlob(data.strip())

header_name = ['labels', 'data']
with open('sentiment1.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=header_name)
    writer.writeheader()

for sent in blob.sentences:
  #print(sent.sentiment.polarity)
  with open('sentiment1.csv', 'a') as file:
    pol=sent.sentiment.polarity
    if(pol<0):
      pol=-1
    else:
      pol=1
    writer = csv.DictWriter(file, fieldnames=header_name)
    info = {
                'labels': pol,
                'data': sent
            }
    writer.writerow(info)

df=[]
import pandas as pd
df=pd.read_csv('sentiment1.csv')

df.head

from nltk.tokenize import word_tokenize
allowed_word_types = ["J"]
all_words=[]
for p in  df['data']:
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    tokenized = word_tokenize(cleaned)
    
    stopped = [w for w in tokenized if not w in stopwords.words('english')]
    
    pos = nltk.pos_tag(stopped)
    
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

"""Making Bag of words model"""

# creating a frequency distribution of each adjectives.
all_words = nltk.FreqDist(all_words)

# listing the 1000 most frequent words
word_features = list(all_words.keys())[:1000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

import random
dataset=[]
i=0
for p in df['data']:
  feature=find_features(p)
  dataset.append([feature,df['labels'][i]])
  i+=1
random.shuffle(dataset)

print(len(dataset))# dataset is a bag of model of vectors

# training_set = dataset[:15000]
# testing_set = dataset[15000:]

"""Applying Machine Learning Algorithm as trained earlier that is SVM"""

predictions=[]
for r in dataset:
  predictions.append(SVC_clf.classify(r[0]))
print(predictions)

ground_truth=[]
for r in dataset:
  ground_truth.append(r[1])

print(str,'classifier accuracy :',accuracy_score(ground_truth,predictions))
print(str,'f1 score :',f1_score(ground_truth,predictions))

"""Sentimental Analysis"""

positive=0
negative=0
for i in range(0,len(predictions)):
  if predictions[i]==1:
    positive=positive+1
  else:
    negative=negative+1

positive_percent=positive/(positive+negative)
negative_percent=negative/(positive+negative)

print(positive_percent*100,negative_percent*100)

"""So this politician (politician Y) has also 94% positive reviews and 6 % negative reviews"""

