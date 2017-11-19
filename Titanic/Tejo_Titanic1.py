# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:23:32 2017

@author: tnuta
"""
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
from matplotlib import pyplot

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
X_train = train_df.drop('Survived',axis=1)
Y_train = train_df['Survived']
X_test  = test_df.drop("PassengerId", axis=1).copy()

train = pd.read_csv("train.csv")
test    = pd.read_csv("test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

namearray= train_df.columns.values
names = []
for i in namearray:
    names.append(i)
    
    
def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = pyplot.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

plot_correlation_map(titanic)
#use seaborn or matplotlib to visualize?
array = np.zeros([2,3])
print(train_df.head())
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#shows high correlation between male and female
#let's check the correlation for the other categories
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#adjust the fare category
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#plot_correlation_map(train_df)

modeldict = {}
logreg = LogisticRegression()
svc = LinearSVC()
randomforest = RandomForestClassifier()
knn = KNeighborsClassifier()
gaussian = GaussianNB
perceptron= Perceptron() # interesting/different
sgd= SGDClassifier()
decisiontree = DecisionTreeClassifier()
linearsvc = LinearSVC()

"""
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

randomforest.fit(X_train, Y_train)
Y_pred = randomforest.predict(X_test)
acc_random_forest = round(randomforest.score(X_train, Y_train) * 100, 2)

perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)


decisiontree.fit(X_train, Y_train)
Y_pred = decisiontree.predict(X_test)
acc_decision_tree = round(decisiontree.score(X_train, Y_train) * 100, 2)


linearsvc.fit(X_train, Y_train)
Y_pred = linearsvc.predict(X_test)
acc_linear_svc = round(linearsvc.score(X_train, Y_train) * 100, 2)
"""
"""
for i in modeldict:
    i.fit(X_train, Y_train)
    """
    

"""
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
"""