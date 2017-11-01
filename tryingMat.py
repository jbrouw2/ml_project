import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np



mat = sio.loadmat('train_subject01.mat')
var_list =sio.whosmat('train_subject01.mat')
x = mat['X']
y = mat['y']
tmin = mat['tmin']
tmax = mat['tmax']
sfreq = mat['sfreq']


def create_features(x,tmin, tmax, sfreq, tmin_origin=- 0.5 ):
    beginning = np.round((tmin-tmin_origin)*sfreq).astype(np.int)
    end = np.round((tmax-tmin_origin)*sfreq).astype(np.int)
    x = x[:,:,beginning:end].copy()

    print "Making 3Ds into 2Ds which is always the preferable amount"
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    print "CS is cancer, and we are also normalizing the data rn"

    x -= x.mean(0)
    x = np.nan_to_num(x / x.std(0))
    return x

print "We will only be doing two subjects today because I'm lame"

Tmin=0.0
Tmax=0.500
sfreq = 250
print " we will be limiting our trials to the time interval [-0.5, 0.500,] because I'm not that stupid"
print "Here we go fam"

X = create_features(x, Tmin, Tmax, sfreq)
Y = y
test_size = 0.33
seed = 10

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


print "your model is in the oven"
model = LogisticRegression()
model.fit(X,Y)


print "accuracy"
print model.score(X_test, Y_test)

print "Answers"
print str(Y_test)
print "Your model's Answers"
print model.predict(X)
