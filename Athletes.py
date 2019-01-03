# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:08:30 2018

@author: Sai Kiran
"""
#DATAREADING

import pandas as pd

df = pd.read_csv('C:\\Users\\vamshi krishna\\Desktop\\athlete_events\\athlete_events_edited.csv')




#SCALING

cleaning = {'Sex':{'F':0,'M':1}}
df.replace(cleaning,inplace=True)

#print(df['Season'].value_counts())

cleaning1 = {'Season':{'Summer':0,'Winter':1}}
df.replace(cleaning1,inplace=True)

#print(df['Medal'].value_counts())

df['Medal'].fillna(0,inplace=True)
cleaning2 = {'Medal':{'Gold':1,'Silver':2,'Bronze':3}}
df.replace(cleaning2,inplace=True)




#LABELENCODING

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(df['Team'])

k=le.transform(df['Team'])

df['Team']=k

le.fit(df['Event'])

j=le.transform(df['Event'])

df['Event']=j




#CORRELATIONANALYSIS

import numpy as np

df1 = pd.read_csv('C:\\Users\\vamshi krishna\\Desktop\\athlete_events\\athlete_events_edited.csv')

df1['Height'].fillna(round(df1['Height'].mean()),inplace=True)

df['Height'].fillna(0,inplace=True)

print(np.corrcoef(df['Height'],df1['Height']))

df1['Weight'].fillna(round(df1['Weight'].median()),inplace=True)

df['Weight'].fillna(0,inplace=True)

print(np.corrcoef(df['Weight'],df1['Weight']))

df1['Age'].fillna(round(df1['Age'].median()),inplace=True)

df['Age'].fillna(0,inplace=True)

print(np.corrcoef(df['Age'],df1['Age']))




#REPLACEMENT

df['Height'].fillna(round(df['Height'].mean()),inplace=True)

df['Weight'].fillna(round(df['Weight'].median()),inplace=True)

df['Age'].fillna(round(df['Age'].median()),inplace=True)




#MODELAPPLICATION

from sklearn import cross_validation, neighbors, linear_model
from sklearn.naive_bayes import GaussianNB

X = np.array(df.drop(['Medal'], 1))
y = np.array(df['Medal'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Knn accuracy'+accuracy)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

accuracy = regr.score(X_test, y_test)
print('Regression accuracy'+accuracy)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

accuracy = gnb.score(X_test, y_test)
print('Naive Bayes accuracy'+accuracy)


