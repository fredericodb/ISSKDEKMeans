import numpy
import random
from scipy.spatial.distance import euclidean, mahalanobis, canberra, cosine, braycurtis, cityblock, chebyshev
from distance_measure import distance_measure, distance_measures, nearest_center
import matplotlib.pyplot as plt
import math
from sklearn.base import BaseEstimator

import time
import logging
from scipy import stats

# Percentage of the authors to use in the initial clustering
PRELIM_DATA_PERCENTAGE = 0.15

# Percentage of each preliminary cluster to use as representative points
REPRESENTATIVE_POINTS_PERCENTAGE = 0.005

# Percentage of distance towards cluster centroid each representative point should travel
CENTROID_MIGRATION_PERCENTAGE = 0.20

# Min distance between CURE clusters without merging
CLUSTER_MERGE_DISTANCE = 0.02

class KMeansCluster:

    def __init__(self):
        self._center = 0
        self._label = 0
        self._points = 0

    def center(self):
        return self._center

    def label(self):
        return self._label

    def points(self):
        return self._points

    def compute_center(self, X):
        self._points = X
        self._center = numpy.mean(X, axis=0)
        return self._center

    def compute_label(self, T):
        yc = stats.mode(T, axis=None)
        # assign modal class to cluster and sample
        if yc.mode.shape[0] == 0:
            # print('Tamanho zero em yc\n')
            self._label = -1
        else:
            self._label = yc.mode[0]
        return self._label

    def compute(self, X, T):
        self.compute_center(X)
        self.compute_label(T)
        return self._center, self._label

class KMeansAlphaCluster(KMeansCluster):

    def __init__(self):
        super().__init__()

    def compute_alpha_center(self, Xm, Xr, alpha=0.75):
        if Xr.shape[0] > 0:
            self._center = alpha * numpy.mean(Xm, axis=0) + (1 - alpha) * numpy.mean(Xr, axis=0)
        else:
            self._center = self.compute_center(Xm)
        return self.center

class ClusteringFeatures(KMeansAlphaCluster):

    def __init__(self, d=1):
        super().__init__()
        self._n = 0
        self._ls = numpy.zeros((d,), dtype=numpy.float64)
        self._ss = numpy.zeros((d,), dtype=numpy.float64)

    def n(self):
        return self._n

    def ls(self):
        return self._ls

    def ss(self):
        return self._ss

    def add_sample(self, x):
        self._n = self._n + 1
        self._ls = self._ls + x
        self._ss = self._ss + x ** 2
        self._center = self._ls / self._n

    def compute_center(self, X):
        self._n = 0
        d = X.shape[1]
        self._ls = numpy.zeros((d,), dtype=numpy.float64)
        self._ss = numpy.zeros((d,), dtype=numpy.float64)
        for x in X:
            self.add_sample(x)
        self._points = X
        return self._center

class CURECluster(ClusteringFeatures):

    def __init__(self, nr=5, pr=0.005):
        super().__init__()
        self._nr = nr
        self._pr = pr
        self._rep_points = []

    def rep_points(self):
        return self._rep_points

    # Computes and stores representative points for this cluster, based on its
    # center and the fixed percentage of points to choose.
    def generate_rep_points(self):
        # Choose the first rep point to be the point furthest point from the "center"
        farthestPoint = None
        farthestDistance = -1
        for x in self._points:
            dist = distance_measure(self._center, x)
            if dist > farthestDistance:
                farthestPoint = x
                farthestDistance = dist
        self._rep_points.append(farthestPoint)
        # numPointsToChoose = int(math.floor(len(X) * REPRESENTATIVE_POINTS_PERCENTAGE))
        numPointsToChoose = self._nr
        # Keep adding points that maximize total distance from each other
        while len(self._rep_points) < numPointsToChoose:
            farthestPoint = None
            farthestTotalDistance = -1
            for x in self._points:
                totalDist = 0
                for rep in self._rep_points:
                    totalDist += distance_measure(rep, x)
                if totalDist > farthestTotalDistance:
                    farthestTotalDistance = totalDist
                    farthestPoint = x
            self._rep_points.append(farthestPoint)

    # Migrates each representative point a fixed percentage towards
    # the centroid of the cluster
    def migrate_rep_points(self):
        for repPoint in self._rep_points:
            for i in range(len(repPoint)):
                distToCenter = math.sqrt((repPoint[i] - self._center[i])**2)
                moveDist = distToCenter * CENTROID_MIGRATION_PERCENTAGE
                if repPoint[i] < self._center[i]:
                    repPoint[i] += moveDist
                elif repPoint[i] > self._center[i]:
                    repPoint[i] -= moveDist

    # Merges this cluster with the given clust, recomputing the centroid
    # and the representative points
    def mergeWithCluster(self, clust):
        self._n = self._n + clust.n()
        self._ls = self._ls + clust.ls()
        self._ss = self._ss + clust.ss()
        self._center = self._ls / self._n
        self._points = numpy.vstack((self._points, clust.points()))
        self.generate_rep_points()
        self.migrate_rep_points()


class ISSCURE(BaseEstimator):

    def __init__(self, nk=2, nr=5, n=2, w=0, dt='euclidean', cini='quantile', nit=100, thd=0, alpha=0.75, mo=1):
        self.clusters = 0
        self.cf_time = 0
        self.cf_weights = 0
        self.cf_distance = 0
        self.nk = nk
        self.nr = nr
        self.n = n
        self.w = w
        self.dt = dt
        self.cini = cini
        self.nit = nit
        self.thd = thd
        self.alpha = alpha
        self.mo = mo
        self.labels_ = 0
        self.accuracy = 0.0
        self.offline_training_time = 0.0

    def fit(self, X, T):
        mode = 'kmeans' # kmeans cure

        if mode == 'cure':
            self.fit_cure(X, T)
        else:
            self.fit_kmeanscure(X, T)

    def cure(self, X, T, nk):
        m = X.shape[0]
        n = X.shape[1]

        ncluster = numpy.zeros((m,))

        # Initialization
        Clusters = []
        numCluster = m
        numPts = m
        distCluster = numpy.ones([m, m])
        distCluster = distCluster * float('inf')
        for idPoint in range(m):
            newClust = CURECluster()
            newClust.compute(X[idPoint, :].reshape((1, X.shape[1])), T[idPoint])
            newClust.generate_rep_points()
            newClust.migrate_rep_points()
            ncluster[idPoint] = idPoint
            Clusters.append(newClust)
        for row in range(0, numPts):
            for col in range(0, row):
                if Clusters[row].label() == Clusters[col].label(): # Isolate different classes leaving dist = infinite
                    distCluster[row][col] = distance_measure(Clusters[row].center(), Clusters[col].center())
        # while (numCluster > nk) and (numpy.min(distCluster) < CLUSTER_MERGE_DISTANCE):
        while numCluster > nk:
            if numpy.mod(numCluster, 50) == 0:
                print('Cluster count:', numCluster)

            # Find a pair of closet clusters
            minIndex = numpy.where(distCluster == numpy.min(distCluster))
            minIndex1 = minIndex[0][0]
            minIndex2 = minIndex[1][0]

            # Merge
            Clusters[minIndex1].mergeWithCluster(Clusters[minIndex2])
            # Update the distCluster matrix
            for i in range(0, minIndex1):
                distCluster[minIndex1, i] = self.get_closest_cluster_to_cluster_dist(Clusters[minIndex1], Clusters[i])
            for i in range(minIndex1+1, numCluster):
                distCluster[i, minIndex1] = self.get_closest_cluster_to_cluster_dist(Clusters[minIndex1], Clusters[i])
            # Delete the merged cluster and its disCluster vector.
            distCluster = numpy.delete(distCluster, minIndex2, axis=0)
            distCluster = numpy.delete(distCluster, minIndex2, axis=1)
            del Clusters[minIndex2]
            numCluster = numCluster - 1

        print('Cluster count:', numCluster)

        return Clusters, numCluster, ncluster

    def fit_cure(self, X, T):
        _start_time = time.time()
        # print('ISSKMeans(nk=%d, n=%d, w=%s, dt=%s, cini=%s, nit=%d, thd=%f, alpha=%f, mo=%d)' % (self.nk, self.n, self.w, self.dt, self.cini, self.nit, self.thd, self.alpha, self.mo))
        print('ISSCURE(nk=%d, n=%d, dt=%s, cini=%s, nit=%d, thd=%f, alpha=%f, mo=%d)' % (self.nk, self.n, self.dt, self.cini, self.nit, self.thd, self.alpha, self.mo))
        nk = self.nk
        m = X.shape[0]
        n = X.shape[1]
        w = self.w[0:nk, :]
        dt = self.dt

        bad_cluster = numpy.zeros((nk,))

        Clusters, numCluster, ncluster = self.cure(X, T, nk)

        # Remove orphan cluster ??????
        # here

        nk = numCluster
        w = self.w[0:nk, :]
        self.cf_time = numpy.zeros((nk, 1))
        self.cf_weights = w

        self.ncluster = ncluster
        self.nk = numCluster
        self.clusters = Clusters

        # Generate sample labels
        output, _ = self.assign_cluster_label(X)

        self.labels_ = numpy.reshape(output, (-1))
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(T, self.labels_)

        # logging.info("number of clusters during learning = %s" % nks[0:r])

        _end_time = time.time()
        self.offline_training_time = (_end_time - _start_time)

        return output


    def fit_kmeanscure(self, X, T):
        _start_time = time.time()
        # print('ISSKMeans(nk=%d, n=%d, w=%s, dt=%s, cini=%s, nit=%d, thd=%f, alpha=%f, mo=%d)' % (self.nk, self.n, self.w, self.dt, self.cini, self.nit, self.thd, self.alpha, self.mo))
        print('ISSCURE(nk=%d, n=%d, dt=%s, cini=%s, nit=%d, thd=%f, alpha=%f, mo=%d)' % (self.nk, self.n, self.dt, self.cini, self.nit, self.thd, self.alpha, self.mo))
        # auto-adding clusters with different influence of samples depending on class
        nk = self.nk
        n = X.shape[1]
        w = self.w[0:nk, :]
        dt = self.dt
        cini = self.cini
        nit = self.nit
        thd = self.thd
        alpha = self.alpha
        mo = self.mo

        # CLUSTER INITIALIZATION
        fcluster = numpy.zeros((nk, n))
        ccluster = numpy.zeros((nk, 1))
        if cini == 'quantile': # initialize clusters
            # distribute examples of each class to clusters equally.
            u = numpy.unique(T)
            nu = u.shape[0]
            iu = 0
            for i in range(0,nk):
                ic = (T == u[iu]).nonzero()
                ic = ic[0]
                nic = ic.shape[0]
                fic = X[ic,:]
                ei = numpy.random.randint(0, nic)
                fcluster[i,:] = fic[ei,:]
                ccluster[i] = u[iu]
                if iu == (nu-1):
                    iu = 0
                else:
                    iu = iu + 1
        elif cini == 'previous': # do not initialize clusters, just read previous ones
            # restore previous clusters
            for i in range(0,nk):
                fcluster[i,:] = self.clusters[i].ls() / self.clusters[i].n()
                ccluster[i] = self.clusters[i].label()
        elif cini == 'point':
            for i in range(0,nk):
                nic = X.shape[0]
                if nic <= 0:
                    print('shit')
                ei = numpy.random.randint(0, nic)
                fcluster[i,:] = X[ei,:]
                ccluster[i] = T[ei]
        else:
            print('inform cini')

        # K-MEANS
        # supervised k-means
        m = X.shape[0]
        if (dt == 'mahalanobis') or (dt == 'weighted mahalanobis'):
            Si = numpy.linalg.inv(numpy.cov(X.T))
        else:
            Si = 0

        ncluster = numpy.zeros((m,))
        bad_cluster = numpy.zeros((nk,))
        r = 1
        t = thd+1
        # maxd = 10;

        nks = numpy.zeros((nit + 2,)) # 1 nk before loop, nit nk during loop, 1 nk after remove bad clusters
        nks[0] = nk

        while (r <= nit) and (t > thd): # maxd revise
            #plt.scatter(X[:, 0], X[:, 1], c="g")

            # calculate distances and assign clusters to samples
            ncluster, _ = nearest_center(fcluster, X, t=dt, Si=Si, w=w)

            bad_cluster = numpy.zeros((nk,))
            # recalculate clusters
            # maxd = 0.0;
            for c in range(0, nk):
                # cluster c codebook
                # cn = fcluster(c,:);
                # obtain samples from cluster c

                # obtain major class samples of cluster c
                from scipy import stats
                iyc = (ncluster == c).nonzero()
                iyc = iyc[0]
                if iyc.shape[0] > 0:
                    # find modal class
                    yc = stats.mode(T[iyc], axis=None)
                    # assign modal class to cluster and sample
                    if yc.mode.shape[0] == 0:
                        logging.debug('interno fcluster ruim %d', c)
                        bad_cluster[c] = 1
                    else:
                        # majority class samples and remaining classes samples contribute to cluster by alpha, 1-alpha
                        imc = (T[iyc] == yc.mode[0]).nonzero()
                        imc = imc[0]
                        irc = (T[iyc] != yc.mode[0]).nonzero()
                        irc = irc[0]
                        if irc.shape[0] > 0:
                            ac = alpha * numpy.mean(X[iyc[imc], :], axis=0) + (1 - alpha) * numpy.mean(X[iyc[irc], :], axis=0)
                            # add new cluster for remaining class samples
                            rc = numpy.mean(X[iyc[irc], :], axis=0)
                            yrc = stats.mode(T[iyc[irc]], axis=None)
                            fcluster = numpy.append(fcluster, [rc], axis=0)
                            ccluster = numpy.append(ccluster, [[yrc.mode[0]]], axis=0)
                            w = numpy.append(w, numpy.ones((1, n)), axis=0)
                            bad_cluster = numpy.append(bad_cluster, [0], axis=0)
                        else:
                            ac = numpy.mean(X[iyc[imc], :], axis=0)
                        # difference of new and old positions
                        t = sum(abs(ac - fcluster[c, :])) / n
                        fcluster[c, :] = ac
                else:
                    logging.debug('interno fcluster ruim %d', c)
                    bad_cluster[c] = 1
            # maybe new clusters
            nk = fcluster.shape[0]

            #plt.scatter(fcluster[:, 0], fcluster[:, 1], c="r")
            nks[r] = nk

            r = r + 1
            #plt.draw()
            #plt.pause(0.0001)

        #plt.show()

        # Marking orphan clusters as bad clusters
        orphan_num = mo
        for c in range(0, nk):
            ioc = (ncluster == c).nonzero()
            ioc = ioc[0]
            if ioc.shape[0] <= orphan_num:
                # orphan sample... sorry
                bad_cluster[c] = 1

        # Eliminate bad clusters
        maskbc = numpy.ones((m,), dtype=bool)
        if (bad_cluster == 1).any():
            # cleaning bad clusters
            ibc = numpy.arange(0, nk)[(bad_cluster == 1)]
            for c in range(0, ibc.shape[0]):
                i = ibc[c]
                iyc = (ncluster == i).nonzero()
                ncluster[iyc] = -1
                maskbc[iyc] = False
            # taking good clusters
            igc = numpy.arange(0, nk)[(bad_cluster == 0)]
            for c in range(0, igc.shape[0]):
                i = igc[c]
                fcluster[c, :] = fcluster[i, :]
                w[c, :] = w[i, :]
                ccluster[c, :] = ccluster[i, :]
                # find samples of cluster i
                iyc = (ncluster == i).nonzero()
                iyc = iyc[0]
                ncluster[iyc] = c
            nk = igc.shape[0]
            fcluster = fcluster[0:nk, :]
            ccluster = ccluster[0:nk, :]
            w = w[0:nk, :]

        nks[r] = nk

        self.cf_time = numpy.zeros((nk, 1))
        self.cf_weights = w
        self.cf_distance = dt
        self.cf_Si = Si
        self.nit = nit

        cure_proc = 'multi' # one class multi
        # Clustering Features & CURE
        # For each CURE cluster, computes the representative points
        # Input: the list of clusters (clusters)
        # Output: the list of clusters, but the clusters now have representative points associated
        #         with themselves (clusters)
        self.clusters = []
        for i in range(0, nk):
            # find samples of cluster u(i)
            iyc = (ncluster == i).nonzero()
            iyc = iyc[0]
            if iyc.shape[0] > 0:
                if cure_proc == 'one':
                    cluster = CURECluster()
                    cluster.compute(X[iyc], T[iyc])
                    cluster.generate_rep_points()
                    cluster.migrate_rep_points()
                    self.clusters.append(cluster)
                elif cure_proc == 'class':
                    yrc = stats.mode(T[iyc], axis=None)
                    iycc = (T[iyc] == yrc.mode[0]).nonzero()
                    iycc = iycc[0]
                    cluster = CURECluster()
                    cluster.compute(X[iyc[iycc]], T[iyc[iycc]])
                    cluster.generate_rep_points()
                    cluster.migrate_rep_points()
                    self.clusters.append(cluster)
                elif cure_proc == 'multi':
                    Clusters, numCluster, _ = self.cure(X[iyc], T[iyc], nk)
                    self.clusters.extend(Clusters)

        self.clusters = self.merge_close_clusters()
        nk = len(self.clusters)
        self.cf_time = numpy.zeros((nk, 1))
        self.cf_weights = numpy.ones((nk, n))

        # Obtain classification
        output, _ = self.assign_cluster_label(X)

        self.ncluster = ncluster
        self.nk = nk
        self.n = n
        # self.clusters = fcluster
        self.labels_ = numpy.reshape(output, (-1))
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(T[maskbc], self.labels_[maskbc])

        # plot results
        #plt.scatter(X[:, 0], X[:, 1], marker="o", c=T)
        #plt.scatter(X[:, 0], X[:, 1], marker="+", c=output[:, 0])
        #plt.show()

        logging.info("number of clusters during learning = %s" % nks[0:r])

        _end_time = time.time()
        self.offline_training_time = (_end_time - _start_time)

        return output

    # Attempts to merge clusters based on the distance between their closest
    # reprsentative points; may result in cluster deletion.
    # Input: the list of all the cluster (clusters)
    # Output: the list of clusters, which may have had clusters merged together. (clusters)
    def merge_close_clusters(self):
        clustersMerged = True
        while clustersMerged:
            for i in range(len(self.clusters)):
                for j in range(i, len(self.clusters)):
                    # if self.clusters[i].label() == self.clusters[j].label(): # for supervised clustering
                        closestDist = self.get_closest_cluster_to_cluster_dist(self.clusters[i], self.clusters[j])
                        if closestDist < CLUSTER_MERGE_DISTANCE:
                            self.clusters[i].mergeWithCluster(self.clusters[j])
                            del self.clusters[j]
                            clustersMerged = True
                            break
                if clustersMerged:
                    clustersMerged = False
                    break
        return self.clusters

    # Helper function for mergeCloseClusters()
    # Determines the closest distance between any two representative points for two clusters
    # Input: Two clusters of type CureCluster (clust1, clust2)
    # Output: the closest distance between any two representative points in the clusters (minDist)
    def get_closest_cluster_to_cluster_dist(self, clust1, clust2):
        minDist = numpy.inf
        for repPoint1 in clust1.rep_points():
            for repPoint2 in clust2.rep_points():
                dist = distance_measure(repPoint1, repPoint2)
                if dist < minDist:
                    minDist = dist
        return minDist

    # Assigns all authors that weren't added via the preliminary clustering
    # to an existing cluster based upon the nearest representative point.
    # Input: the list of clusters, the dictionary of authors, and the dictionary of authors
    #        involved in the initial clustering.
    # Output: An updated list of clusters, which now contains all the authors in them. (clusters)
    def assign_cluster_label(self, X):
        output = -numpy.ones((X.shape[0], 1))
        dist = numpy.zeros((X.shape[0],), dtype=numpy.float64)
        for i in range(X.shape[0]):
            clust, d = self.get_closest_cluster(X[i])
            if clust == None:
                print("empty cluster")
                continue
            output[i] = clust.label()
            dist[i] = d
        return output, dist

    # Helper function for assignRemainingData()
    # Determines the cluster associated with the representative point closest
    # to the given author.
    # Input: a given author of class Author, and the list of clusters (author, clusters)
    # Output: the clostest cluster for the author, based on repPoints (clustChoice)
    def get_closest_cluster(self, xi):
        clustChoice = None
        minDist = numpy.inf
        for cluster in self.clusters:
            for repPoint in cluster.rep_points():
                dist = distance_measure(xi, repPoint)
                if dist < minDist:
                    minDist = dist
                    clustChoice = cluster
        return clustChoice, minDist

    def predict_with_dist(self, X):
        output, dist = self.assign_cluster_label(X)
        return output, dist

    def predict(self, X):
        output, _ = self.predict_with_dist(X)
        return output

    # Assigns all authors that weren't added via the preliminary clustering
    # to an existing cluster based upon the nearest representative point.
    # Input: the list of clusters, the dictionary of authors, and the dictionary of authors
    #        involved in the initial clustering.
    # Output: An updated list of clusters, which now contains all the authors in them. (clusters)
    def assign_cluster(self, X):
        clusters = []
        dist = numpy.zeros((X.shape[0],), dtype=numpy.float64)
        for i in X.shape[0]:
            clust, d = self.get_closest_cluster(X[i])
            clusters.append(clust)
            dist[i] = d
        return clusters, dist

    def get_closest_clusters(self, xi):
        clustChoices = [None] * self.nk
        minDists = numpy.inf * numpy.ones((self.nk,), dtype=numpy.float64)
        i = 0
        for cluster in self.clusters:
            for repPoint in cluster.rep_points():
                dist = distance_measure(xi, repPoint)
                if dist < minDists[i]:
                    minDists[i] = dist
                    clustChoices[i] = cluster
            i = i + 1
        return clustChoices, minDists

    def fit_labeled(self, xi, yi, wd=0, vector=False):

        cs, ds = self.get_closest_clusters(xi)

        # PROXIMITY / SIMILARITY MEASURES
        # select high confidence unlabeled examples
        # 1) if any closest two clusters have the same class as xi
        nnc = numpy.argsort(ds)
        ds = ds[nnc]
        i = 0
        c1 = -1
        while (c1 != yi) and (i < nnc.shape[0]):
            c1 = self.clusters[nnc[i]].label()  # class of cluster 1
            d1 = ds[i]
            i = i + 1

        if c1 == yi:
            nncluster = nnc[i - 1]
        else:
            nncluster = -1
        # print("crbf: %d" % crbf)
        # print("nnc[0]: %d" % nnc[0])
        # plt.scatter(ds,rbfs)
        # plt.show()

        n = self.n
        w1 = 1
        w2 = 0

        # Clustering Features
        # update Clustering Features
        ncls = int(nncluster)
        if ncls != -1:
            ci = self.clusters[ncls].label()
            if wd == 0: # full sample is added
                new_cluster = CURECluster()
                new_X = numpy.reshape(xi, (1, self.n))
                new_T = numpy.reshape(ci, (1, 1))
                new_cluster.compute(new_X, new_T)
                self.clusters[ncls].mergeWithCluster(new_cluster)

                self.cf_time[ncls] = 0

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 1: # linear weight
                new_cluster = CURECluster()
                new_X = numpy.reshape(xi, (1, self.n))
                new_T = numpy.reshape(ci, (1, 1))
                new_cluster.compute(new_X, new_T)
                self.clusters[ncls].mergeWithCluster(new_cluster)

                self.cf_time[ncls] = 0

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nnc[1]
                    new_cluster = CURECluster()
                    new_X = numpy.reshape(xi, (1, self.n))
                    new_T = numpy.reshape(ci, (1, 1))
                    new_cluster.compute(new_X, new_T)
                    self.clusters[ncls].mergeWithCluster(new_cluster)

                    self.cf_time[ncls] = 0

            if wd == 2: # radial basis function
                ncls = nrbf[0]
                #w1 = rbfs[nrbf[0]]
                #w2 = rbfs[nrbf[1]]
                w1 = 1
                w2 = 0
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w1 * xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w1 * xi) ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nrbf[1]
                    self.cf_n[ncls] = self.cf_n[ncls] + 1
                    self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w2 * xi
                    self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w2 * xi) ** 2
                    self.cf_time[ncls] = 0
                    self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

            if wd == 3: # distance is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                d = (xi - self.clusters[ncls, :])
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + d
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + d ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))
        else: # create new cluster for new example
            ci = c1
            new_cluster = CURECluster()
            new_X = numpy.reshape(xi, (1, self.n))
            new_T = numpy.reshape(ci, (1, 1))
            new_cluster.compute(new_X, new_T)
            self.clusters.append(new_cluster)

            self.cf_time = numpy.append(self.cf_time, 0)
            self.cf_weights = numpy.append(self.cf_weights, numpy.ones((1, n)), axis=0)

            # merge and remove clusters
            # ttime = i;
            # tm = mod(ttime, 10);
            # if (tm == 0)
            #   [CF, ncluster] = check_merging_clusters(CF, ncluster, 0.5);
            # [CF, ncluster] = remove_clusters(CF, ncluster, check_for_removal_clusters(CF, 1, 10, 1), 0);

    def fit_unlabeled(self, xi, wd=0, vector=False):

        cs, ds = self.get_closest_clusters(xi)

        # PROXIMITY / SIMILARITY MEASURES
        # select high confidence unlabeled examples
        # 1) if closest clusters have the same class
        nnc = numpy.argsort(ds)

        c1 = self.clusters[nnc[0]].label() # class of cluster 1
        d1 = ds[nnc[0]]
        if nnc.shape[0] > 1:
            c2 = self.clusters[nnc[1]].label() # class of cluster 2
            d2 = ds[nnc[1]]
        else:
            c2 = c1
            d2 = d1

        dsv = numpy.zeros((self.nk, self.n))
        if vector:
            x1 = self.clusters[nnc[0]].center()
            d1 = distance_measure(x1, xi, t='deltas')
            if nnc.shape[0] > 1:
                x2 = self.clusters[nnc[1]].center()
                d2 = distance_measure(x2, xi, t='deltas')
            else:
                d2 = d1

            for c in range(0, self.nk):
                xc = self.clusters[nnc[c]].center()
                dsv[c, :] = distance_measure(xc, xi, t='deltas')

        # 2) if distance to the nearest cluster is within R
        Rs = numpy.zeros((self.nk,))
        Rv = numpy.zeros((self.nk, self.n))
        if not vector:
            # R1 = numpy.sqrt(sum(self.cf_weights[nnc[0], :]*(self.clusters[nnc[0]].ss() - (self.clusters[nnc[0]].ls()**2)/self.clusters[nnc[0]].n())/self.clusters[nnc[0]].n()))
            # if nnc.shape[0] > 1:
            #     R2 = numpy.sqrt(sum(self.cf_weights[nnc[1], :]*(self.clusters[nnc[1]].ss() - (self.clusters[nnc[1]].ls()**2)/self.clusters[nnc[1]].n())/self.clusters[nnc[1]].n()))
            # else:
            #     R2 = R1

            for c in range(0, self.nk):
                Rs[c] = numpy.sqrt(sum(self.cf_weights[nnc[c], :]*(self.clusters[nnc[c]].ss() - (self.clusters[nnc[c]].ls()**2)/self.clusters[nnc[c]].n())/self.clusters[nnc[c]].n()))
        else:
            R1 = numpy.sqrt(self.cf_weights[nnc[0], :] * (self.clusters[nnc[0]].ss() - (self.clusters[nnc[0]].ls() ** 2) / self.clusters[nnc[0]].n()) / self.clusters[nnc[0]].n())
            if nnc.shape[0] > 1:
                R2 = numpy.sqrt(self.cf_weights[nnc[1], :] * (self.clusters[nnc[1]].ss() - (self.clusters[nnc[1]].ls() ** 2) / self.clusters[nnc[1]].n()) / self.clusters[nnc[1]].n())
            else:
                R2 = R1

            for c in range(0, self.nk):
                Rs[c] = numpy.sqrt(sum(self.cf_weights[nnc[c], :]*(self.clusters[nnc[c]].ss() - (self.clusters[nnc[c]].ls()**2)/self.clusters[nnc[c]].n())/self.clusters[nnc[c]].n()))

            for c in range(0, self.nk):
                Rv[c, :] = numpy.sqrt(self.cf_weights[nnc[c], :] * (self.clusters[nnc[c]].ss() - (self.clusters[nnc[c]].ls() ** 2) / self.clusters[nnc[c]].n()) / self.clusters[nnc[c]].n())

        n = self.n
        w1 = 1
        w2 = 0
        mr = 3
        # # se dois clusters tem classes iguais
        # if c1 == c2:
        #     # se dentro do raio dos dois
        #     if (d1 < R1).all() & (d2 < R2).all():
        #         nncluster = nnc[0]
        #         w1 = (1 - d1 / (d1 + d2))
        #         w2 = (1 - d2 / (d1 + d2))
        #         w1 = 1
        #         w2 = 1
        #     # se dentro do 3 x raio dos dois
        #     elif (d1 < mr * R1).all() & (d2 < mr * R2).all():
        #         nncluster = nnc[0]
        #         w1 = (1 - d1 / (d1 + d2))
        #         w2 = (1 - d2 / (d1 + d2))
        #         w1 = 1
        #         w2 = 1
        #     # se dentro do raio do mais próximo
        #     elif (d1 < R1).all():
        #         nncluster = nnc[0]
        #         w1 = 1
        #         w2 = 0
        #     # se dentro do raio do segundo mais próximo
        #     elif (nnc.shape[0] > 1) and (d2 < R2).all():
        #         nncluster = nnc[1]
        #         w1 = 0
        #         w2 = 1
        #     else:
        #         nncluster = -1
        # # se dois clusters tem classes diferentes
        # else:
        #     # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
        #     if (d2 > d1).all() & (d1 < R1).all() & (d2 > R2).any():
        #         nncluster = nnc[0]
        #         w1 = (1 - d1 / (d1 + d2))
        #         w2 = 0
        #         w1 = 1
        #     # se distância do segundo é maior que 2 x a distância do primeiro e dentro do 3 x raio do primeiro e fora do raio do segundo
        #     elif (d2 > d1).all() & (d1 < mr * R1).all() & (d2 > R2).any():
        #         nncluster = nnc[0]
        #         w1 = (1 - d1 / (d1 + d2))
        #         w2 = 0
        #         w1 = 1
        #     # se dentro do raio do mais próximo
        #     elif (d1 < R1).all():
        #         nncluster = nnc[0]
        #         w1 = 1
        #         w2 = 0
        #     # se dentro do raio do segundo mais próximo
        #     elif (nnc.shape[0] > 1) and (d2 < R2).all():
        #         nncluster = nnc[1]
        #         w1 = 0
        #         w2 = 1
        #     else:
        #         nncluster = -1

        # Similarity measure evaluation sorting
        # distance / radius
        mr = 0.90 # 0.75
        DR = ds / Rs
        # # gaussian rbf
        # mr = 0.8 # 0.75
        # DR = 1 - (1 / (numpy.sqrt(2 * math.pi) * Rs)) * numpy.exp(- 0.5 * (ds / Rs) ** 2)
        # # semicircle
        # mr = 0.4
        # DR = (Rs - Rs * numpy.sin(numpy.arccos(ds / Rs))) / Rs
        # # distance / sqrt(radius)
        # mr = 1.25 # 0.75
        # DR = ds / numpy.sqrt(Rs)
        # # distance / radius^2
        # mr = 1.25 # 0.75
        # DR = ds / (Rs ** 2)
        # Sorting and filtering
        nnc = numpy.argsort(DR)
        DRmask = (DR <= mr)
        DRsel = (DRmask[nnc]).nonzero()
        DRsel = DRsel[0]
        nnc = nnc[DRsel]

        if nnc.size == 1:
            nncluster = nnc[0]
            w1 = 1
            w2 = 0
        elif nnc.size >= 2:
            nncluster = nnc[0]
            if self.clusters[nnc[0]].label() == self.clusters[nnc[1]].label():
                w1 = 1
                w2 = 0
            else:
                i = numpy.random.randint(0, 1, 1)
                if i == 0:
                    nncluster = nnc[0]
                    w1 = 1
                    w2 = 0
                else:
                    nncluster = nnc[1]
                    w1 = 0
                    w2 = 1
                nncluster = nnc[0]
        else:
            nncluster = -1

        #
        # # 3) RBF
        # if wd == 2:  # radial basis function
        #     if vector:
        #         ds = numpy.zeros((self.nk,self.n))
        #     else:
        #         ds = numpy.zeros((self.nk,))
        #     rbfs = numpy.zeros((self.nk,))
        #     for c in range(0, self.nk):
        #         xc = self.clusters[c, :]
        #         if vector:
        #             dd = numpy.sqrt((xc - xi) ** 2)
        #             R = numpy.sqrt(self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c])
        #         else:
        #             dd = distance_xcluster(xc, xi, self.cf_distance, self.cf_Si, self.cf_weights[c, :])
        #             R = numpy.sqrt(sum(self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c]))
        #
        #         rbfs[c] = sum((1 / (numpy.sqrt(2 * math.pi) * R)) * numpy.exp(- 0.5 * (dd / R) ** 2))
        #         #rbfs[c] = math.exp(-dd ** 2)
        #         if vector:
        #             ds[c, :] = dd
        #         else:
        #             ds[c] = dd
        #
        #     nrbf = numpy.argsort(rbfs)
        #     nrbf = numpy.flipud(nrbf)

        # print("crbf: %d" % crbf)
        # print("nnc[0]: %d" % nnc[0])
        # plt.scatter(ds,rbfs)
        # plt.show()

        # Clustering Features
        # update Clustering Features
        ncls = int(nncluster)
        if ncls != -1:
            ci = self.clusters[ncls].label()
            if wd == 0: # full sample is added
                new_cluster = CURECluster()
                new_X = numpy.reshape(xi, (1, self.n))
                new_T = numpy.reshape(ci, (1, 1))
                new_cluster.compute(new_X, new_T)
                self.clusters[ncls].mergeWithCluster(new_cluster)

                self.cf_time[ncls] = 0

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 1: # linear weight
                if w1 != 0:
                    ncls = nnc[0]
                    new_cluster = CURECluster()
                    new_X = numpy.reshape(xi, (1, self.n))
                    new_T = numpy.reshape(ci, (1, 1))
                    new_cluster.compute(new_X, new_T)
                    self.clusters[ncls].mergeWithCluster(new_cluster)

                    self.cf_time[ncls] = 0

                    self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nnc[1]
                    new_cluster = CURECluster()
                    new_X = numpy.reshape(xi, (1, self.n))
                    new_T = numpy.reshape(ci, (1, 1))
                    new_cluster.compute(new_X, new_T)
                    self.clusters[ncls].mergeWithCluster(new_cluster)

                    self.cf_time[ncls] = 0

                    self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 2: # radial basis function
                ncls = nrbf[0]
                #w1 = rbfs[nrbf[0]]
                #w2 = rbfs[nrbf[1]]
                w1 = 1
                w2 = 0
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls,:] = self.cf_ls[ncls,:] + w1 * xi
                self.cf_ss[ncls,:] = self.cf_ss[ncls,:] + (w1 * xi) ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls,:] = self.cf_ls[ncls,:] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nrbf[1]
                    self.cf_n[ncls] = self.cf_n[ncls] + 1
                    self.cf_ls[ncls,:] = self.cf_ls[ncls,:] + w2 * xi
                    self.cf_ss[ncls,:] = self.cf_ss[ncls,:] + (w2 * xi) ** 2
                    self.cf_time[ncls] = 0
                    self.clusters[ncls,:] = self.cf_ls[ncls,:] / self.cf_n[ncls]

            if wd == 3: # distance is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                d = (xi - self.clusters[ncls,:])
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + d
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + d ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            # merge and remove clusters
            # ttime = i;
            # tm = mod(ttime, 10);
            # if (tm == 0)
            #   [CF, ncluster] = check_merging_clusters(CF, ncluster, 0.5);
            # [CF, ncluster] = remove_clusters(CF, ncluster, check_for_removal_clusters(CF, 1, 10, 1), 0);

        return c1, (ncls != -1)


class ISSCUREEnsemble(BaseEstimator):
    def __init__(self, nk=2, nr=5, n=2, w=0, dt='euclidean', cini=True, nit=100, thd=0, alpha=0.75, mo=1, ver=2):
        self.nk = nk
        self.nr = nr
        self.n = n
        self.w = w
        self.dt = dt
        self.cini = cini
        self.nit = nit
        self.thd = thd
        self.alpha = alpha
        self.mo = mo
        self.ver = ver
        # self.wkm = numpy.zeros((self.nkm,), dtype=numpy.float64)
        self.mode = 1 # 0: vote; 1: distance
        self.offline_training_time = 0.0

    def fit(self, X, y):
        _start_time = time.time()
        self.n = X.shape[1]
        self.nkm = self.n+1
        self.kms = []
        self.fsel = numpy.ones((self.nkm, self.n), dtype=bool)
        indices = []
        for i in range(self.n):
            index = random.randint(0, self.n - 1)
            while index in indices:
                index = random.randint(0, self.n - 1)
            indices.append(index)
            self.fsel[i, index] = False

        pout = numpy.zeros((self.nkm, X.shape[0]), dtype=numpy.int64)
        for i in range(0, self.nkm):
            self.kms.append(ISSCURE(self.nk, self.nr, sum(self.fsel[i, :]), self.w[:, self.fsel[i, :]], self.dt, self.cini, self.nit, self.thd, self.alpha, self.mo, self.ver))
            pout[i, :] = self.kms[i].fit(X[:, self.fsel[i, :]], y).reshape(-1)
        _end_time = time.time()
        self.offline_training_time = _end_time - _start_time

        from scipy import stats
        mode = stats.mode(pout, axis=0)
        return mode[0].reshape(-1)

    def fit_labeled(self, xi, yi):
        for i in range(0, self.nkm):
            self.kms[i].fit_labeled(xi[:, self.fsel[i, :]], yi)

    def fit_unlabeled(self, xi):
        pucls = numpy.zeros((self.nkm,), dtype=numpy.int64)
        puacpt = numpy.zeros((self.nkm,), dtype=bool)
        pudist = numpy.zeros((self.nkm,), dtype=numpy.float64)
        for i in range(0, self.nkm):
            pucls[i], puacpt[i], pudist[i] = self.kms[i].fit_unlabeled2_with_dist(xi[self.fsel[i, :]])

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
        pout = numpy.zeros((self.nkm, X.shape[0]), dtype=numpy.int64)
        pdist = numpy.zeros((self.nkm, X.shape[0]), dtype=numpy.float64)
        for i in range(0, self.nkm):
            out, dist = self.kms[i].predict_with_dist(X[:, self.fsel[i, :]])
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

