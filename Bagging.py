from datetime import datetime

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics import log_loss


class Bagger(object):
    def __init__(self, clf, clf_params=None, nbags=10, seed=0, regression=False,
                 subsample=1., bootstrap=False, shuffle=False, metric=log_loss, verbose=True):
        self.clf = clf
        self.clf_params = clf_params if clf_params is not None else {}
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.shuffle = shuffle
        self.metric = metric
        self.verbose = verbose

    def fit(self, x_train, y_train, x_test=None, y_test=None, x_val = None, y_val = None):

        if not (isinstance(x_train, np.ndarray) or isinstance(x_train, spmatrix)):
            x_train = np.array(x_train)

        if not (isinstance(x_test, np.ndarray) or isinstance(x_test, spmatrix)):
            x_test = np.array(x_test)

        cnt = len(x_train)

        samples = np.random.choice(cnt, int(cnt * self.subsample),self.bootstrap)
        for i in range(self.nbags):
            clf = self.clf()
            clf.fit(x_train[samples,:],y_train[samples])





    @property
    def test_predictions(self):
        return self.yhat_test

    @property
    def probe_predictions(self):
        return self.yhat_probe



from sklearn.ensemble import RandomForestClassifier

bg = Bagger(RandomForestClassifier,clf_params={"n_estimators":10},subsample=0.8)
a = np.random.rand(10,20)
b = np.random.randint(0,3,10)

print(np.shape(a))
print(np.shape(b))
bg.fit(a,b)