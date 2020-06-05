import numpy
from distance_measure import distance_measures
from scipy import stats

class KNN:
    def KNN(self, k=1):
        self.k = k

    def predict(self, X, T, xi):
        ds = distance_measures(X, xi.reshape((1, xi.reshape(-1).shape[0])))
        ds = ds.reshape(-1)

        nnc = numpy.argsort(ds)
        P = []
        for i in range(self.k):
            P.append(T[nnc[i]])  # class label

        y = stats.mode(P, axis=None)
        return y.mode[0]