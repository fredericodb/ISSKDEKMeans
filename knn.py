import numpy
from distance_measure import distance_measures
from scipy import stats


class KNN:
    def __init__(self, k=1):
        self.k = k
        self.X = 0
        self.T = 0

    def fit(self, X, T):
        self.X = X
        self.T = T

    def predict(self, xi):
        ds = distance_measures(self.X, xi.reshape((1, xi.reshape(-1).shape[0])))
        ds = ds.reshape(-1)

        nnc = numpy.argsort(ds)
        P = []
        for i in range(self.k):
            P.append(self.T[nnc[i]])  # class label

        y = stats.mode(P, axis=None)
        return y.mode[0]
