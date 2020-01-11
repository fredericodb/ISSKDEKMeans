import numpy
from distance_xcluster import distance_xcluster
import matplotlib.pyplot as plt
import math

import logging

class Seeded_KMeans:

    def __init__(self, nk, n):
        self.clusters = 0
        self.nk = nk
        self.n = n
        self.labels_ = 0
        self.accuracy = 0.0
        self.ncluster = 0
        self.ccluster = 0

    def find_class_groups(self, X, T, nk):
        S = numpy.zeros((nk, X.shape[1]))
        TS = numpy.zeros((nk,))
        u = numpy.unique(T)
        nu = u.shape[0]
        n_c = int(nk / nu)
        k = 0
        for i in range(0, nu):
            ic = (T == u[i])
            Xi = X[ic, :]

            if n_c > 1:
                from sklearn.cluster import KMeans
                kmu = KMeans(init='random', n_clusters=n_c, n_init=1)
                kmu.fit(Xi)

            for j in range(0, n_c):
                if Xi.shape[0] >= n_c:
                    S[k, :] = kmu.cluster_centers_[j, :]
                else:
                    S[k, :] = Xi[j, :]
                TS[k] = u[i]
                k = k + 1

        return S, TS

    def fit(self, X, T, nit, thd):
        nk = self.nk
        n = self.n
        fcluster = numpy.zeros((nk, n))
        ccluster = numpy.zeros((nk,))
        # initialize clusters
        S, TS = self.find_class_groups(X, T, nk)
        # cluster centers are equal the mean of same class examples
        for i in range(0, nk):
            fcluster[i, :] = S[i, :]
            ccluster[i] = TS[i]

        # supervised k-means
        m = X.shape[0]
        ncluster = numpy.zeros((m,))
        r = 1
        t = thd+1
        # maxd = 10;
        while (r <= nit) | (t > thd): # maxd revise
            #plt.scatter(X[:, 0], X[:, 1], c="g")
            # calculate distances and assign clusters to samples
            for i in range(0, m):
                xi = X[i, :]
                mind = -1
                for c in range(0, nk):
                    xc = fcluster[c,:]
                    dd = numpy.sqrt(sum(numpy.power(xc - xi, 2)))
                    # dd = sqrt(sum((xc - xi).^2))
                    if (mind == -1) | (dd < mind):
                        mind = dd
                        ncluster[i] = c

            # recalculate clusters
            for c in range(0, nk):
                # cluster c codebook
                # obtain samples from cluster c
                ic = (ncluster == c).nonzero()
                ic = ic[0]
                if ic.shape[0] > 0:
                    # calculate mean of samples in cluster c
                    ac = numpy.mean(X[ic, :], axis=0)
                    # difference of new and old positions
                    t = sum(abs(ac - fcluster[c, :]))/n
                    # formatSpec = 'dd = %1.4f \n'
                    # fprintf(formatSpec, dd)
                    # update cluster c
                    fcluster[c, :] = ac

            #plt.scatter(fcluster[:, 0], fcluster[:, 1], c="r")
            r = r + 1
            #plt.draw()
            #plt.pause(0.0001)
        #plt.show()

        # obtain classification
        output = numpy.zeros((m,1))
        from scipy import stats
        for i in range(0,nk):
            # find samples of cluster u(i)
            iyc = (ncluster == i).nonzero()
            iyc = iyc[0]
            if (iyc.shape[0] > 0):
                output[iyc] = ccluster[i]
            else:
                logging.debug('output fcluster ruim %d', i)

        self.ncluster = ncluster
        self.nk = nk
        self.clusters = fcluster
        self.cclusters = ccluster
        self.labels_ = output
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(T, self.labels_)

        # plot results
        #plt.scatter(X[:, 0], X[:, 1], marker="o", c=T)
        #plt.scatter(X[:, 0], X[:, 1], marker="+", c=output[:, 0])
        #plt.show()

        return output

    def predict(self, X):
        m = X.shape[0]
        nk = self.nk
        output = numpy.zeros((m, 1))
        nclstr = numpy.zeros((m, 1))
        ds = numpy.zeros((nk,))
        # calculate distances and assign clusters to samples
        for i in range(0,m):
            xi = X[i,:]
            mind = -1
            for c in range(0,nk):
                xc = self.clusters[c,:]
                dd = numpy.sqrt(sum(numpy.power(xc - xi, 2)))
                # dd = sqrt(sum((w. * (xc - xi)). ^ 2));
                if (mind == -1) | (dd < mind):
                    mind = dd
                    minc = c

                ds[c] = dd

            nclstr[i] = minc
            output[i] = self.ccluster[minc]

        return output