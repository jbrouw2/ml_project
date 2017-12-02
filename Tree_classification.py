import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

import numpy as np # linear algebra
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from subprocess import check_output

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
combine = [train, test]

#print ('# features:')
#print train.columns.values
#print '_'*40
#print ('# features:')
#print test.columns.values

Y = train['Cover_Type']
X = train.drop(['Cover_Type'], axis=1)

model = LogisticRegression()
model.fit(X, Y)
#print model.score(X, Y)
#print model.predict(test)

Soil_1_5 = train['Soil_Type1']+ train['Soil_Type2'] + train['Soil_Type3']+ train['Soil_Type4'] + train['Soil_Type5']
#print Soil_1_5

sns.boxplot(x="Cover_Type", y="Elevation", data=train);
plt.show()

for dataset in combine:
    dataset.loc[ dataset['Elevation'] <= 16, 'Elevation'] = 0
    dataset.loc[(dataset['Elevation'] > 16) & (dataset['Elevation'] <= 32), 'Elevation'] = 1
    dataset.loc[(dataset['Elevation'] > 32) & (dataset['Elevation'] <= 48), 'Elevation'] = 2
    dataset.loc[(dataset['Elevation'] > 48) & (dataset['Elevation'] <= 64), 'Elevation'] = 3
    dataset.loc[ dataset['Elevation'] > 64, 'Elevation'] = 4


#print train[['Soil_Type1','Cover_Type']].groupby(['Soil_Type1'], as_index=False).mean()
#print train[['Soil_Type2','Cover_Type']].groupby(['Soil_Type2'], as_index=False).mean()
#
#print train[['Soil_Type3','Cover_Type']].groupby(['Soil_Type3'], as_index=False).mean()
#print train[['Soil_Type4','Cover_Type']].groupby(['Soil_Type4'], as_index=False).mean()