# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:16:12 2017

@author: Home
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

#reading the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting = 3 )
ps = PorterStemmer()
#Cleaning the data in review column
corpus = []
for i in range(0,1000):
  #  review = dataset['Review'][6]
    review = re.sub('[^a-z A-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('English')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Splitting to training and test datasets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Applying feature scaling to x-dataset
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(x_train,y_train)
y_pred = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)

from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier2.fit(x_train,y_train)
y_pred2 = classifier2.predict(x_test)
cm2 = confusion_matrix(y_test,y_pred2)

from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(criterion='entropy', random_state = 0)
classifier3.fit(x_train,y_train)
y_pred3 = classifier3.predict(x_test)
cm3 = confusion_matrix(y_test,y_pred3)