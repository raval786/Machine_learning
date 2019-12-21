# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:25:08 2019

@author: Raval
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Restaurant_reviews.tsv", delimiter = '\t', quoting = 3) # in file line seperated with tab not comma and it's a tsv file so pandas expected csv file so we have to add delimiter parameter
# quoting = 3 means we ignoring the quotes

# Cleaning the text
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # corpus is collection of text(anything)
for i in range(0, 1000): # 999 are data but upper bound is exlude so we take 1000
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # re.sub removes all the pancuation marks like comma,double quotes,dot dot it removes all
    # first time dataset['Review'][0] take one review when it's fully clean the apply for loop
    # and after '[^a-zA-Z]' , ' ' we put space vecause we don't want to our words will join together after removing the puntuation marks so we put space.
    # dataset['Review'][0] we take the first review of our dataset.
    review = review.lower() # convert all captial letteres to lower
    review = review.split() # it split the string into words
    # Stemming
    # stemming is taking the root of the word like loved it convert into love because it meaning or sentimate is same
    # if we don't stemming after threr are lots of inrelavent word like love, loved ,loves and all means the same so when we have to do sparce matrix if will be very difficult
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]# stopwords is all the unnecessary words like the,place,is,that it removes. so ml algorithm can coreleate with best and suitable words
    # we use set() because in large dataset like ariticles it hepls us to work fasetr. words('english') because here we use english comments.
    # ps.stem(word) means it steamms the words like convert loved into love.
    review = ' '.join(review) # we join seperate words in whole word but we the to not combine all 3 words we have to put space and then join the words.
    corpus.append(review)
    
# Create a bag of words model (create sparse matrix)
    
# Tokenisation means take the words from all the dataset. but words is not the duplicate so it create the column cantains of all the words and allthe rows containg all the review. so it create matrix. this is the bag of words model.
#  with CountVectorizer we can clean the text without run that above for loop. but if we have to cleaning the complex data like scrap the html page. so it cantains very complex text.so that time we can use above code.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # before we don't apply max_feature it has 1565 columns it is words.but what max_feature doing is this that it remove the words that comes 1 or 2 times .so becareful to choose max_feature because if choose to small then our model will be spoiled 
# so in the xall the columns are words in the review like wow, love etc. and row is that it contains the review. words.if yes then 1 no to 0. 
X = cv.fit_transform(corpus).toarray() # .toarray() is for create matrix. it is independent variable
y = dataset.iloc[:,1].values # it is dependent variable. it contains the result of the review whether it's positive or negative.


# APPLYING THE NAIVE BAYES ALGORITHM

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)
