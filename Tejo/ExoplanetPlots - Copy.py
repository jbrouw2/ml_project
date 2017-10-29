
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

filename = 'exoTest.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:,1:]
Y = array[:,0]
kfold = KFold(n_splits=100, random_state=7)
model = KNeighborsClassifier()
scoring = 'accuracy'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

model2 = KNeighborsRegressor()
results2 = cross_val_score(model2, X, Y, cv=kfold, scoring='r2')
print(results2.mean())


