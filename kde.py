import numpy
import random
import time
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import math

from sklearn.base import BaseEstimator

import logging

class KDE(BaseEstimator):

    def __init__(self, kernel='gaussian', bandwidth=-1):
        # bins: int, sequence, string(method)
        self.kernel = kernel
        self.bandwidth = bandwidth

    def get_kernel_density(self, X):
        # Grid search best bandwidth using Cross-Validation
        if self.bandwidth == -1:
            if X.shape[0] > 10:
                from sklearn.model_selection import GridSearchCV
                grid = GridSearchCV(KernelDensity(),
                                    {'bandwidth': numpy.linspace(0.1, 1.0, 30)},
                                    cv=10)  # 10-fold cross-validation
                grid.fit(X)
                bandwidth = grid.best_params_['bandwidth']
            else:
                bandwidth = 0.2

        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(X)
        pdf = kde_skl

        return pdf

    def fit(self, X, T):
        T_unique = sorted(numpy.unique(T))
        clusters = []
        classes = []
        logpriors = []
        for t in T_unique:
            t_samples = (T == t)
            pdf = self.get_kernel_density(X[t_samples, :])
            clusters.append(pdf)
            classes.append(t)
            logpriors.append(numpy.log(X[t_samples, :].shape[0] / X.shape[0]))
        self.clusters = clusters
        self.nk = len(clusters)
        self.classes = classes
        self.X = X
        self.T = T
        self.logpriors = logpriors

        return self.predict(X)

    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
        logprobs = numpy.vstack([pdf.score_samples(X)
                              for pdf in self.clusters]).T
        result = numpy.exp(logprobs + self.logpriors)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
        m = X.shape[0]
        n = X.shape[1]
        proba = self.predict_proba(X).T
        y = [self.classes[i] for i in numpy.argmax(proba, axis=0)]

        return y

    def fit_labeled(self, xi, yi):
        xi = xi.reshape(-1)
        xi = xi.reshape((1, xi.shape[0]))
        yi = yi.reshape((1,))
        # inefficient way
        self.X = numpy.concatenate((self.X, xi), axis=0)
        self.T = numpy.concatenate((self.T, yi))
        self.fit(self.X, self.T)
        return self.predict(xi)[0]

    def fit_unlabeled(self, xi):
        # inefficient way
        yi = self.predict(xi)[0]
        return self.fit_labeled(xi, yi), True

    def fit_unlabeled_batch(self, X):
        return 0

class KDEEnsemble(BaseEstimator):
    def __init__(self, kernel='gaussian', bandwidth=-1):
        self.kernel = kernel
        self.bandwidth = bandwidth
        # self.wkm = numpy.zeros((self.nkm,), dtype=numpy.float64)
        self.mode = 1 # 0: vote; 1: distance
        self.offline_training_time = 0.0

    def fit(self, X, y):
        _start_time = time.time()
        self.n = X.shape[1]
        self.nmodels = self.n+1
        self.models = []
        self.fsel = numpy.ones((self.nmodels, self.n), dtype=bool)
        indices = []
        for i in range(self.n):
            index = random.randint(0, self.n - 1)
            while index in indices:
                index = random.randint(0, self.n - 1)
            indices.append(index)
            self.fsel[i, index] = False

        pout = numpy.zeros((self.nmodels, X.shape[0]), dtype=numpy.int64)
        for i in range(0, self.nmodels):
            self.models.append(KDE())
            pout[i, :] = self.models[i].fit(X[:, self.fsel[i, :]], y).reshape(-1)
        _end_time = time.time()
        self.offline_training_time = _end_time - _start_time

        from scipy import stats
        mode = stats.mode(pout, axis=0)
        return mode[0].reshape(-1)

    def fit_labeled(self, xi, yi):
        for i in range(0, self.nmodels):
            self.models[i].fit_labeled(xi[:, self.fsel[i, :]], yi)

    def fit_unlabeled(self, xi):
        pucls = numpy.zeros((self.nmodels,), dtype=numpy.int64)
        puacpt = numpy.zeros((self.nmodels,), dtype=bool)
        pudist = numpy.zeros((self.nmodels,), dtype=numpy.float64)
        for i in range(0, self.nmodels):
            pucls[i], puacpt[i], pudist[i] = self.models[i].fit_unlabeled(xi[self.fsel[i, :]])

        puclsacpt = pucls[puacpt]
        if self.mode == 0:
            from scipy import stats
            mode = stats.mode(puclsacpt)
            if len(puclsacpt) > 0:
                return mode[0][0], True
            else:
                return -1, False
        else:
            if len(puclsacpt) > 0:
                pudist = pudist[puacpt]
                neari = numpy.argmin(pudist)
                if isinstance(neari, numpy.ndarray):
                    return puclsacpt[neari[0]], True
                else:
                    return puclsacpt[neari], True
            else:
                return -1, False


    def predict(self, X):
        pout = numpy.zeros((self.nmodels, X.shape[0]), dtype=numpy.int64)
        pdist = numpy.zeros((self.nmodels, X.shape[0]), dtype=numpy.float64)
        for i in range(0, self.nmodels):
            out, dist = self.models[i].predict(X[:, self.fsel[i, :]])
            pout[i, :], pdist[i, :] = out.reshape(-1), dist.reshape(-1)

        if self.mode == 0:
            from scipy import stats
            mode = stats.mode(pout, axis=0)
            return mode[0].reshape(-1)
        else:
            neari = numpy.argmin(pdist, axis=0)
            if isinstance(neari, numpy.ndarray):
                return pout[neari[0], :].reshape(-1)
            else:
                return pout[neari, :].reshape(-1)
