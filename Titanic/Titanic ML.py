import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
testA = pd.read_csv('gender_submission.csv')
combine = [train, test]

#for dataset in combine:
 #   dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
  #  dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
   # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    #dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    #dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in combine:
    dataset['Age'] = np.nan_to_num(dataset['Age'])
#^^^changing NaN values from NaN to zero

#Uncomment these to get some data details
#print ('# features:')
#print train.columns.values
#print '_'*40
#print ('# data types:')
#print train.info()
#print '_'*40
#print test.info()
#print train.describe()

#print train.describe(include=['O'])
#^^^This describes all categorical data (where there is 1s and 0s



def chance_of_survival_given_category(x):
    return train[[x,'Survived']]\
    .groupby([x])\
    .mean()\
    .sort_values(by='Survived',ascending =False)

# ^^^This function takes a variable (by name) and shows survival rate for every value in the set
# only use this for categorical data

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#^^^isolating titles


#print pd.crosstab(train['Title'], train['Sex'])
#^^^Prints titles and their concentration by gender

titles = {"Mr":1, "Miss":2,"Mrs":3, "Master":4, "Rare":5}
#^^^changing titles into numerical values


for dataset in combine:
     dataset['Title'] = dataset['Title'].map(titles)
     dataset['Title'] = dataset['Title'].fillna(0)

#^^^Adding numerical titles to the data set
#print train.head()

#print train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
#^^^shows titles by survival rate

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male': 0}).astype(int)
#^^^changing gender into binary variable female=1 male=0
#print train.head()

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':1,'C':2,'Q':3})
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
#^^^changing embarked into numerical values
#print train.head()

#graph = sns.FacetGrid(train, col='Survived')
#graph.map(plt.hist,'Age', bins=20)
#^^^some simple visualizations

x_train=train.drop(['PassengerId','Survived', 'Ticket','Cabin', 'Name'], axis=1)
y_train = train['Survived']
x_test=test.drop(['PassengerId', 'Ticket','Cabin', 'Name',], axis=1)
y_test= testA['Survived']
X = x_train
Y = y_train
#^^^making test and train sets

#Looking over test and train sets
#print x_test.head()
#print '_'*40
#print x_train.head()
#print '_'*40
#print y_train.head()
#print '_'*40
#print y_test.head()

model = LogisticRegression()
model.fit(X, Y)
print model.score(x_test, y_test)
print model.predict(x_test)