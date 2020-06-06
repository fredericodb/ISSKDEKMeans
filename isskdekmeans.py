import numpy
import random
from kde import KDE
from distance_measure import distance_measure, distance_measures, nearest_center
import matplotlib.pyplot as plt
import math
from sklearn.base import BaseEstimator

import time
import logging


class ISSKDEKMeans(BaseEstimator):

    def __init__(self, nk=2, n=2, w=0, dt='euclidean', cini='quantile', nit=100, thd=0, alpha=0.75, mo=1, use_kde=True,
                 kde_kernel='gaussian', mr=3):
        self.clusters = 0
        self.cf_n = 0
        self.cf_ls = 0
        self.cf_ss = 0
        self.cf_time = 0
        self.cf_class = 0
        self.cf_weights = 0
        self.cf_distance = 0
        self.nk = nk
        self.n = n
        self.w = w
        self.dt = dt
        self.cini = cini
        self.nit = nit
        self.thd = thd
        self.alpha = alpha
        self.mo = mo
        self.use_kde = use_kde
        self.kde_kernel = kde_kernel
        self.labels_ = 0
        self.accuracy = 0.0
        self.offline_training_time = 0.0
        self.itconv = nit
        self.kde_class = 0
        self.kde_cluster = 0
        self.kde_clusterclass = []
        self.kde_better = False
        self.kde_clusterbetter = []
        self.mr = mr

    def fit(self, X, T):
        _start_time = time.time()
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
        fcluster = numpy.zeros((nk, n))
        ccluster = numpy.zeros((nk, 1))
        if cini == 'quantile':  # initialize clusters
            # distribute examples of each class to clusters equally.
            u = numpy.unique(T)
            nu = u.shape[0]
            iu = 0
            for i in range(0, nk):
                ic = (T == u[iu]).nonzero()
                ic = ic[0]
                nic = ic.shape[0]
                fic = X[ic, :]
                ei = numpy.int(numpy.random.uniform(0, nic, (1, 1)))
                fcluster[i, :] = fic[ei, :]
                ccluster[i] = u[iu]
                if iu == (nu - 1):
                    iu = 0
                else:
                    iu = iu + 1
        elif cini == 'previous':  # do not initialize clusters, just read previous ones
            # restore previous clusters
            for i in range(0, nk):
                fcluster[i, :] = self.cf_ls[i, :] / self.cf_n[i]
                ccluster[i] = self.cf_class[i]
        elif cini == 'point':
            for i in range(0, nk):
                nic = X.shape[0]
                ei = numpy.int(numpy.random.uniform(0, nic, (1, 1)))
                fcluster[i, :] = X[ei, :]
                ccluster[i] = T[ei]
        else:
            print('inform cini')

        # supervised k-means
        m = X.shape[0]
        if (dt == 'mahalanobis') or (dt == 'weighted mahalanobis'):
            Si = numpy.linalg.inv(numpy.cov(X.T))
        else:
            Si = 0
        ncluster = numpy.zeros((m,))
        bad_cluster = numpy.zeros((nk,))
        r = 1
        t = thd + 1
        # maxd = 10;

        nks = numpy.zeros((nit + 2,))  # 1 nk before loop, nit nk during loop, 1 nk after remove bad clusters
        nks[0] = nk

        while (r <= nit) and (t > thd):  # maxd revise
            # plt.scatter(X[:, 0], X[:, 1], c="g")

            # calculate distances and assign clusters to samples
            ncluster, _ = nearest_center(fcluster, X, t=dt, Si=Si, w=w)

            # for i in range(0, m):
            #     xi = X[i, :]
            #     mind = -1
            #     for c in range(0, nk):
            #         xc = fcluster[c, :]
            #         dd = distance_measure(xc, xi, t=dt, Si=Si, w=w[c, :])
            #         # dd = sqrt(sum((w. * (xc - xi)). ^ 2))
            #         if (mind == -1) | (dd < mind):
            #             mind = dd
            #             ncluster[i] = c

            bad_cluster = numpy.zeros((nk,))
            # recalculate clusters
            Tc = []
            ak = 0
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
                        mrc = len(imc) + len(irc)
                        wmc = len(imc) / mrc
                        wrc = len(irc) / mrc
                        if irc.shape[0] > 0:
                            ac = alpha * wmc * numpy.mean(X[iyc[imc], :], axis=0) + (1 - alpha) * wrc * numpy.mean(
                                X[iyc[irc], :], axis=0)
                            # ac = alpha * numpy.mean(X[iyc[imc], :], axis=0) - (1 - alpha) * numpy.mean(X[iyc[irc], :], axis=0)

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
                        tc = sum(abs(ac - fcluster[c, :])) / n
                        Tc.append(tc)
                        fcluster[c, :] = ac
                else:
                    logging.debug('interno fcluster ruim %d', c)
                    bad_cluster[c] = 1
            # threshold
            t = numpy.max(Tc)
            # maybe new clusters
            nk = fcluster.shape[0]

            # plt.scatter(fcluster[:, 0], fcluster[:, 1], c="r")
            nks[r] = nk

            r = r + 1
            # plt.draw()
            # plt.pause(0.0001)

        # plt.show()

        self.itconv = r

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

        # Clustering Features
        self.cf_n = numpy.zeros((nk, 1))
        self.cf_ls = numpy.zeros((nk, n))
        self.cf_ss = numpy.zeros((nk, n))
        self.cf_time = numpy.zeros((nk, 1))
        self.cf_class = numpy.zeros((nk, 1))
        self.cf_weights = w
        self.cf_distance = dt
        self.cf_Si = Si
        self.nit = nit

        for i in range(0, m):
            xi = X[i, :]
            # update Clustering Features
            ncls = int(ncluster[i])
            # test ncls of out of bounds indexes
            if ncls == -1:
                logging.debug('Hey! you were in a bad cluster %d', ncls)
            else:
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + xi ** 2
                self.cf_class[ncls] = ccluster[ncls]

        # obtain classification
        output_kms = -numpy.ones((m, 1))
        from scipy import stats
        for i in range(0, nk):
            # find samples of cluster u(i)
            iyc = (ncluster == i).nonzero()
            iyc = iyc[0]
            if iyc.shape[0] > 0:
                # find modal class
                yc = stats.mode(T[iyc], axis=None)
                # assign modal class to cluster and sample
                if yc.mode.shape[0] == 0:
                    # print('Tamanho zero em yc\n')
                    self.cf_class[i] = -1
                    output_kms[iyc] = -1
                else:
                    self.cf_class[i] = yc.mode[0]
                    output_kms[iyc] = yc.mode[0]
            else:
                logging.debug('output fcluster ruim %d', i)

        self.ncluster = ncluster
        self.nk = nk
        self.n = n
        self.clusters = fcluster
        self.labels_ = output_kms
        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(T[maskbc], self.labels_[maskbc])
        self.kde_clusterbetter = numpy.zeros((nk,), dtype=bool)

        # plot results
        # plt.scatter(X[:, 0], X[:, 1], marker="o", c=T)
        # plt.scatter(X[:, 0], X[:, 1], marker="+", c=output[:, 0])
        # plt.show()

        logging.info("number of clusters during learning = %s" % nks[0:r])

        if self.use_kde:
            # Kernel Density Estimator for classes and clusters
            kde = KDE()
            kde.fit(X, T)
            self.kde_class = kde
            kde = KDE()
            kde.fit(X[maskbc, :], ncluster[maskbc])
            self.kde_cluster = kde
            self.kde_clusterclass = []
            output_kde = numpy.zeros((m, 1))
            for i in range(0, nk):
                # find samples of cluster u(i)
                iyc = (ncluster == i).nonzero()
                iyc = iyc[0]
                if iyc.shape[0] > 0:
                    kde = KDE()
                    kde.fit(X[iyc], T[iyc])
                    self.kde_clusterclass.append(kde)
                    output_kde[iyc, 0] = kde.predict(X[iyc])
                    accuracy_kde = accuracy_score(T[iyc], output_kde[iyc, 0])
                    accuracy_kms = accuracy_score(T[iyc], output_kms[iyc, 0])
                    if accuracy_kde >= accuracy_kms:
                        self.labels_[iyc] = output_kde[iyc]
                        self.kde_clusterbetter[i] = True
                else:
                    logging.debug('output fcluster ruim %d', i)

            # accuracy_kde = accuracy_score(T[maskbc], output_kde[maskbc])
            # if accuracy_kde > self.accuracy:
            #     self.accuracy = accuracy_kde
            #     self.labels_ = output_kde
            #     self.kde_better = True
            #     output = output_kde

        output = self.labels_
        new_accuracy = accuracy_score(T[maskbc], output[maskbc])
        if new_accuracy > self.accuracy:
            logging.info('Improved accuracy from %f -> %f' % (self.accuracy, new_accuracy))
        self.accuracy = new_accuracy

        _end_time = time.time()
        self.offline_training_time = (_end_time - _start_time)
        return output

    def predict_with_dist(self, X):
        m = X.shape[0]
        nk = self.nk
        w = self.cf_weights
        # output = numpy.zeros((m, 1))
        # nclstr = numpy.zeros((m, 1))
        # dist = numpy.zeros((m, 1))
        # ds = numpy.zeros((nk,))

        # calculate distances and assign clusters to samples
        nclstr, dist = nearest_center(self.clusters, X, t=self.cf_distance, Si=self.cf_Si, w=w)
        output = self.cf_class[nclstr]

        # for i in range(0,m):
        #     xi = X[i,:]
        #     mind = -1
        #     maxp = -1
        #     for c in range(0,nk):
        #         xc = self.clusters[c,:]
        #         dd = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=w[c,:])
        #         # dd = sqrt(sum((w. * (xc - xi)). ^ 2));
        #         if (mind == -1) | (dd < mind):
        #             mind = dd
        #             minc = c
        #
        #         ds[c] = dd
        #
        #         # prob = self.kde_clusterclass[c].predict_proba(xi)
        #         # prob = prob[0]
        #         # iprob = numpy.argmax(prob)
        #         # if (maxp == -1) or (prob[iprob] > maxp).all():
        #         #     maxp = prob[iprob]
        #         #     maxpc = c
        #
        #     nclstr[i] = minc
        #     dist[i] = mind
        #     output[i] = self.cf_class[minc]

        # KDE predictions for some clusters
        for i in range(0, m):
            if self.use_kde:
                if self.kde_clusterbetter[nclstr[i]]:  # previously if self.kde_better:
                    xi = X[i, :]
                    py = self.kde_clusterclass[nclstr[i]].predict(xi)
                    try:
                        output[i] = self.kde_clusterclass[nclstr[i]].predict(xi)
                    except ValueError:
                        print("hey")

                # # vote for class
            # cand = numpy.zeros((3,))
            # cand[0] = self.cf_class[minc]
            # cand[1] = self.kde_class.predict(xi)[0]
            # cand[2] = self.kde_clusterclass[maxpc].predict(xi)[0]
            #
            # from scipy import stats
            # mode = stats.mode(cand, axis=0)
            # output[i] = mode[0].reshape(-1)

        return output, dist

    def predict(self, X):
        output, _ = self.predict_with_dist(X)
        return output

    def fit_labeled(self, xi, yi, wd=0, vector=False):
        ds = numpy.zeros((self.nk,))
        mind = -1
        for c in range(0, self.nk):
            xc = self.clusters[c, :]
            dd = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
            # dd = sqrt(sum((w. * (xc - xi)). ^ 2));
            if (mind == -1) | (dd < mind):
                mind = dd
                minc = c

            ds[c] = dd

        # PROXIMITY / SIMILARITY MEASURES
        # select high confidence unlabeled examples
        # 1) if any closest two clusters have the same class as xi
        nnc = numpy.argsort(ds)
        ds = ds[nnc]
        c1 = self.cf_class[nnc[0]]  # class of cluster 1
        d1 = ds[0]
        if nnc.shape[0] > 1:
            c2 = self.cf_class[nnc[1]]  # class of cluster 2
            d2 = ds[1]
        else:
            c2 = c1
            d2 = d1

        if vector:
            x1 = self.clusters[nnc[0], :]
            d1 = distance_measure(x1, xi)
            if nnc.shape[0] > 1:
                x2 = self.clusters[nnc[1], :]
                d2 = distance_measure(x2, xi)
            else:
                d2 = d1

        if not vector:
            R1 = numpy.sqrt(sum(self.cf_weights[nnc[0], :] * (
                        self.cf_ss[nnc[0], :] - (self.cf_ls[nnc[0], :] ** 2) / self.cf_n[nnc[0]]) / self.cf_n[nnc[0]]))
            if nnc.shape[0] > 1:
                R2 = numpy.sqrt(sum(self.cf_weights[nnc[1], :] * (
                            self.cf_ss[nnc[1], :] - (self.cf_ls[nnc[1], :] ** 2) / self.cf_n[nnc[1]]) / self.cf_n[
                                        nnc[1]]))
            else:
                R2 = R1
        else:
            R1 = numpy.sqrt(self.cf_weights[nnc[0], :] * (
                        self.cf_ss[nnc[0], :] - (self.cf_ls[nnc[0], :] ** 2) / self.cf_n[nnc[0]]) / self.cf_n[nnc[0]])
            if nnc.shape[0] > 1:
                R2 = numpy.sqrt(self.cf_weights[nnc[1], :] * (
                            self.cf_ss[nnc[1], :] - (self.cf_ls[nnc[1], :] ** 2) / self.cf_n[nnc[1]]) / self.cf_n[
                                    nnc[1]])
            else:
                R2 = R1

        if (yi == c1) and (d1 <= R1).all():
            nncluster = nnc[0]
        elif (yi == c2) and (d2 <= R2).all():
            nncluster = nnc[1]
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
            if wd == 0:  # full sample is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + xi ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 1:  # linear weight
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w1 * xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w1 * xi) ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nnc[1]
                    self.cf_n[ncls] = self.cf_n[ncls] + 1
                    self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w2 * xi
                    self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w2 * xi) ** 2
                    self.cf_time[ncls] = 0
                    self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

            if wd == 2:  # radial basis function
                ncls = nrbf[0]
                # w1 = rbfs[nrbf[0]]
                # w2 = rbfs[nrbf[1]]
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

            if wd == 3:  # distance is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                d = (xi - self.clusters[ncls, :])
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + d
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + d ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))
        else:  # create new cluster for new example
            self.cf_n = numpy.append(self.cf_n, 1)
            self.cf_ls = numpy.append(self.cf_ls, numpy.reshape(xi, (1, xi.shape[0])), axis=0)
            self.cf_ss = numpy.append(self.cf_ss, numpy.reshape(xi ** 2, (1, xi.shape[0])), axis=0)
            self.cf_time = numpy.append(self.cf_time, 0)
            self.cf_class = numpy.append(self.cf_class, numpy.reshape(yi, (1, 1), axis=0))
            kde = KDE()
            kde.fit(xi, yi)
            self.kde_clusterclass.append(kde)
            self.kde_clusterbetter = numpy.append(self.kde_clusterbetter, True)
            self.clusters = numpy.append(self.clusters, numpy.reshape(xi, (1, xi.shape[0])), axis=0)
            ncls = self.cf_n.shape[0]
            self.ncluster = numpy.concatenate((self.ncluster, [ncls]))
            self.cf_weights = numpy.append(self.cf_weights, numpy.ones((1, n)), axis=0)

            # merge and remove clusters
            # ttime = i;
            # tm = mod(ttime, 10);
            # if (tm == 0)
            #   [CF, ncluster] = check_merging_clusters(CF, ncluster, 0.5);
            # [CF, ncluster] = remove_clusters(CF, ncluster, check_for_removal_clusters(CF, 1, 10, 1), 0);

    def fit_unlabeled(self, xi, wd=0, vector=False):

        ds = distance_measures(self.clusters, xi.reshape((1, xi.reshape(-1).shape[0])), t=self.cf_distance,
                               Si=self.cf_Si, w=self.cf_weights)
        ds = ds.reshape(-1)
        # ds = numpy.zeros((self.nk,))
        # mind = -1
        # for c in range(0, self.nk):
        #     xc = self.clusters[c, :]
        #     dd = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
        #     if (mind == -1) | (dd < mind):
        #         mind = dd
        #         minc = c
        #
        #     ds[c] = dd

        # PROXIMITY / SIMILARITY MEASURES
        # select high confidence unlabeled examples
        # 1) if closest clusters have the same class
        nnc = numpy.argsort(ds)
        ds = ds[nnc]
        c1 = self.cf_class[nnc[0]]  # class of cluster 1
        d1 = ds[0]
        if nnc.shape[0] > 1:
            c2 = self.cf_class[nnc[1]]  # class of cluster 2
            d2 = ds[1]
        else:
            c2 = c1
            d2 = d1

        dsv = numpy.zeros((self.nk, self.n))
        if vector:
            x1 = self.clusters[nnc[0], :]
            d1 = distance_measure(x1, xi, t='deltas')
            if nnc.shape[0] > 1:
                x2 = self.clusters[nnc[1], :]
                d2 = distance_measure(x2, xi, t='deltas')
            else:
                d2 = d1

            for c in range(0, self.nk):
                xc = self.clusters[nnc[c], :]
                dsv[c, :] = distance_measure(xc, xi, t='deltas')

        # 2) if distance to the nearest cluster is within R
        Rs = numpy.zeros((self.nk,))
        Rv = numpy.zeros((self.nk, self.n))
        if not vector:
            R1 = numpy.sqrt(sum(self.cf_weights[nnc[0], :] * (
                        self.cf_ss[nnc[0], :] - (self.cf_ls[nnc[0], :] ** 2) / self.cf_n[nnc[0]]) / self.cf_n[nnc[0]]))
            if nnc.shape[0] > 1:
                R2 = numpy.sqrt(sum(self.cf_weights[nnc[1], :] * (
                            self.cf_ss[nnc[1], :] - (self.cf_ls[nnc[1], :] ** 2) / self.cf_n[nnc[1]]) / self.cf_n[
                                        nnc[1]]))
            else:
                R2 = R1
        else:
            R1 = numpy.sqrt(self.cf_weights[nnc[0], :] * (
                        self.cf_ss[nnc[0], :] - (self.cf_ls[nnc[0], :] ** 2) / self.cf_n[nnc[0]]) / self.cf_n[nnc[0]])
            if nnc.shape[0] > 1:
                R2 = numpy.sqrt(self.cf_weights[nnc[1], :] * (
                            self.cf_ss[nnc[1], :] - (self.cf_ls[nnc[1], :] ** 2) / self.cf_n[nnc[1]]) / self.cf_n[
                                    nnc[1]])
            else:
                R2 = R1

            for c in range(0, self.nk):
                Rs[c] = numpy.sqrt(sum(self.cf_weights[nnc[c], :] * (
                            self.cf_ss[nnc[c], :] - (self.cf_ls[nnc[c], :] ** 2) / self.cf_n[nnc[c]]) / self.cf_n[
                                           nnc[c]]))

            for c in range(0, self.nk):
                Rv[c, :] = numpy.sqrt(self.cf_weights[nnc[c], :] * (
                            self.cf_ss[nnc[c], :] - (self.cf_ls[nnc[c], :] ** 2) / self.cf_n[nnc[c]]) / self.cf_n[
                                          nnc[c]])

        n = self.n
        w1 = 1
        w2 = 0
        mr = self.mr
        # se dois clusters tem classes iguais
        if c1 == c2:
            # se dentro do raio dos dois
            if (d1 < R1).all() & (d2 < R2).all():
                nncluster = nnc[0]
                w1 = (1 - d1 / (d1 + d2))
                w2 = (1 - d2 / (d1 + d2))
            # se dentro do 3 x raio dos dois
            elif (d1 < mr * R1).all() & (d2 < mr * R2).all():
                nncluster = nnc[0]
                w1 = (1 - d1 / (d1 + d2))
                w2 = (1 - d2 / (d1 + d2))
            # se dentro do raio do mais próximo
            elif (d1 < R1).all():
                nncluster = nnc[0]
                w1 = 1
                w2 = 0
            # se dentro do raio do segundo mais próximo
            elif (nnc.shape[0] > 1) and (d2 < R2).all():
                nncluster = nnc[1]
                w1 = 1
                w2 = 0
            else:
                nncluster = -1
        # se dois clusters tem classes diferentes
        else:
            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
            if (d2 > d1).all() & (d1 < R1).all() & (d2 > R2).any():
                nncluster = nnc[0]
                w1 = (1 - d1 / (d1 + d2))
                w2 = 0
            # se distância do segundo é maior que 2 x a distância do primeiro e dentro do 3 x raio do primeiro e fora do raio do segundo
            elif (d2 > d1).all() & (d1 < mr * R1).all() & (d2 > R2).any():
                nncluster = nnc[0]
                w1 = (1 - d1 / (d1 + d2))
                w2 = 0
            # se dentro do raio do mais próximo
            elif (d1 < R1).all():
                nncluster = nnc[0]
                w1 = 1
                w2 = 0
            # se dentro do raio do segundo mais próximo
            elif (nnc.shape[0] > 1) and (d2 < R2).all():
                nncluster = nnc[1]
                w1 = 1
                w2 = 0
            else:
                nncluster = -1

        if self.use_kde:
            nncluster_kde = int(self.kde_cluster.predict(xi)[0])
            if (nnc[0] == nncluster_kde) or (nnc[1] == nncluster_kde):
                nncluster = nncluster_kde
                c1 = numpy.array(self.kde_clusterclass[nncluster].predict(xi))
            else:
                c1_kde = self.kde_class.predict(xi)[0]
                if c1 != c1_kde:
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
            if wd == 0:  # full sample is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + xi ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 1:  # linear weight
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w1 * xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w1 * xi) ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nnc[1]
                    self.cf_n[ncls] = self.cf_n[ncls] + 1
                    self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w2 * xi
                    self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w2 * xi) ** 2
                    self.cf_time[ncls] = 0
                    self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

            if wd == 2:  # radial basis function
                ncls = nrbf[0]
                # w1 = rbfs[nrbf[0]]
                # w2 = rbfs[nrbf[1]]
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

            if wd == 3:  # distance is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                d = (xi - self.clusters[ncls, :])
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

    def fit_unlabeled2_with_dist(self, xi, wd=0):
        ds = numpy.zeros((self.nk,))
        dv = numpy.zeros((self.nk, self.n))
        Rs = numpy.zeros((self.nk,))
        Rv = numpy.zeros((self.nk, self.n))
        dcos = numpy.zeros((self.nk,))
        dman = numpy.zeros((self.nk,))
        dcan = numpy.zeros((self.nk,))
        dbc = numpy.zeros((self.nk,))
        davg = numpy.zeros((self.nk,))
        dcze = numpy.zeros((self.nk,))
        ddiv = numpy.zeros((self.nk,))
        dche = numpy.zeros((self.nk,))
        dcor = numpy.zeros((self.nk,))

        for c in range(0, self.nk):
            xc = self.clusters[c, :]
            ds[c] = distance_measure(xc, xi, t='euclidean', w=self.cf_weights[c, :])
            dcos[c] = distance_measure(xc, xi, t='cosine')
            dman[c] = distance_measure(xc, xi, t='cityblock')
            dcan[c] = distance_measure(xc, xi, t='canberra')
            dbc[c] = distance_measure(xc, xi, t='braycurtis')
            davg[c] = distance_measure(xc, xi, t='average')
            dcze[c] = distance_measure(xc, xi, t='czekanowski')
            ddiv[c] = distance_measure(xc, xi, t='divergence')
            dche[c] = distance_measure(xc, xi, t='chebyshev')
            dcor[c] = distance_measure(xc, xi, t='correlation')
            Rs[c] = numpy.sqrt(
                sum(self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c]))
            dv[c, :] = distance_measure(xc, xi, t='deltas')
            Rv[c, :] = numpy.sqrt(
                self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c])

        # Multi-objective optimization: \epsilon -constraint method
        f = numpy.zeros((13,), dtype=numpy.float64)
        fv = numpy.zeros((2, self.n), dtype=numpy.float64)
        wf = numpy.array([0.6, 0.0, 0.4])
        nobj = 13
        objf = numpy.zeros((self.nk, nobj), dtype=numpy.float64)
        for c in range(0, self.nk):
            f[0] = ds[c] / numpy.max(ds)
            f[1] = dcos[c] / numpy.max(dcos)
            f[2] = dman[c] / numpy.max(dman)
            f[3] = dcan[c] / numpy.max(dcan)
            f[4] = dbc[c] / numpy.max(dbc)
            f[5] = davg[c] / numpy.max(davg)
            f[6] = dcze[c] / numpy.max(dcze)
            f[7] = ddiv[c] / numpy.max(ddiv)
            f[8] = dche[c] / numpy.max(dche)
            f[9] = dcor[c] / numpy.max(dcor)
            f[10] = (ds[c] / Rs[c]) / numpy.max(ds / Rs)
            fv[0, :] = dv[c, :] / numpy.max(dv, axis=0)
            fv[1, :] = (dv[c, :] / Rv[c, :]) / numpy.max(dv / Rv, axis=0)
            nanf = numpy.isnan(f)
            f[nanf] = 1
            nanf = numpy.isnan(fv)
            fv[nanf] = 1
            f[11] = numpy.mean(fv[0, :])
            f[12] = numpy.mean(fv[1, :])
            objf[c, :] = f

            # if (dsv[c, :] < mr * Rv[c, :]).all():
            #     nncluster = nnc[c]
            #     break

        rejf = 0.4
        nncluster = -1
        c1 = -1
        dist = -1

        # Pareto front reordered
        from pareto_efficiency import is_pareto_efficient_dumb
        minf = numpy.argsort(objf[:, 0])
        robjf = objf[minf, :]
        is_pareto_front = is_pareto_efficient_dumb(robjf)
        if is_pareto_front is False:
            print('problem')
        else:
            if is_pareto_front.size > 0:
                winner = minf[is_pareto_front][0]
            else:
                winner = minf[is_pareto_front]
        win = 0
        nwin = len(minf[is_pareto_front])
        while (objf[winner, :] >= rejf).any():
            win = win + 1
            if win <= (nwin - 1):
                winner = minf[is_pareto_front][win]
            else:
                break
        if (objf[winner, :] < rejf).all():
            nncluster = winner
            c1 = self.cf_class[winner]
            dist = objf[winner, 0]

        # while (c < self.nk) and (objf[c] < rejf):
        #
        #     nncluster = minf[c]
        #     dist = objf[c]
        #
        #     # vote for class
        #     cand = numpy.zeros((3,))
        #     cand[0] = self.cf_class[minf[c]]
        #     cand[1] = self.kde_class.predict(xi)[0]
        #     cand[2] = self.kde_clusterclass[maxpc].predict(xi)[0]
        #
        #     from scipy import stats
        #     mode = stats.mode(cand, axis=0)
        #     c1 = mode[0].reshape(-1)
        #     if c1 == cand[2]:
        #         nncluster = maxpc
        #
        #     c = c + 1

        # print("crbf: %d" % crbf)
        # print("nnc[0]: %d" % nnc[0])
        # plt.scatter(ds,rbfs)
        # plt.show()

        # Clustering Features
        # update Clustering Features
        ncls = int(nncluster)
        if ncls != -1:
            if wd == 0:  # full sample is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + xi ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

            if wd == 1:  # linear weight
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w1 * xi
                self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w1 * xi) ** 2
                self.cf_time[ncls] = 0
                self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

                self.ncluster = numpy.concatenate((self.ncluster, [ncls]))

                if w2 != 0:
                    ncls = nnc[1]
                    self.cf_n[ncls] = self.cf_n[ncls] + 1
                    self.cf_ls[ncls, :] = self.cf_ls[ncls, :] + w2 * xi
                    self.cf_ss[ncls, :] = self.cf_ss[ncls, :] + (w2 * xi) ** 2
                    self.cf_time[ncls] = 0
                    self.clusters[ncls, :] = self.cf_ls[ncls, :] / self.cf_n[ncls]

            if wd == 2:  # radial basis function
                ncls = nrbf[0]
                # w1 = rbfs[nrbf[0]]
                # w2 = rbfs[nrbf[1]]
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

            if wd == 3:  # distance is added
                self.cf_n[ncls] = self.cf_n[ncls] + 1
                d = (xi - self.clusters[ncls, :])
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

        return c1, (ncls != -1), dist

    def fit_unlabeled2(self, xi, wd=0):
        c1, accept, _ = self.fit_unlabeled2_with_dist(xi, wd)
        return c1, accept

    def fit_unlabeled_batch(self, X, w, dt, cini, nit, thd, cd):
        nk = self.nk
        n = self.n
        fcluster = numpy.zeros((nk, n))
        ccluster = numpy.zeros((nk, 1))
        if cini:  # initialize clusters
            # select cluster centroids randomly among samples
            m = X.shape[0]
            for i in range(0, nk):
                ei = numpy.int(numpy.random.uniform(0, m, (1, 1)))
                fcluster[i, :] = X[ei, :]
        else:  # do not initialize clusters, just read previous ones
            # restore previous clusters
            for i in range(0, nk):
                fcluster[i, :] = self.clusters[i, :]
                ccluster[i] = self.cf_class[i]

        # unsupervised k-means
        m = X.shape[0]
        if (dt == 'mahalanobis') or (dt == 'weighted mahalanobis'):
            Si = numpy.linalg.inv(numpy.cov(X.T))
        else:
            Si = 0
        ncluster = numpy.zeros((m,))
        bad_cluster = numpy.zeros((nk,))
        r = 1
        t = thd + 1
        # maxd = 10;
        while (r <= nit) | (t > thd):  # maxd revise
            # plt.scatter(X[:, 0], X[:, 1], c="g")
            # calculate distances and assign clusters to samples
            for i in range(0, m):
                xi = X[i, :]
                mind = -1
                for c in range(0, nk):
                    xc = fcluster[c, :]
                    dd = distance_measure(xc, xi, t=dt, Si=Si, w=w[c, :])
                    # dd = sqrt(sum((w. * (xc - xi)). ^ 2))
                    if (mind == -1) | (dd < mind):
                        mind = dd
                        ncluster[i] = c

            bad_cluster = numpy.zeros((nk,))
            # recalculate clusters
            # maxd = 0.0;
            for c in range(0, nk):
                # cluster c codebook
                # cn = fcluster(c,:);
                # obtain samples from cluster c

                ic = (ncluster == c).nonzero()
                ic = ic[0]
                if ic.shape[0] > 0:
                    # calculate mean of samples in cluster c
                    ac = numpy.mean(X[ic, :], axis=0)
                    # difference of new and old positions
                    t = sum(abs(ac - fcluster[c, :])) / n
                    # distance from cluster c to mean
                    # dd = distance_xc(dt, cn, ac, S, w)
                    # formatSpec = 'dd = %1.4f \n'
                    # fprintf(formatSpec, dd)
                    # if (dd > maxd):
                    # maxd = dd
                    # update cluster c
                    fcluster[c, :] = ac
                else:
                    logging.debug('interno fcluster ruim %d', c)
                    bad_cluster[c] = 1

            # plt.scatter(fcluster[:, 0], fcluster[:, 1], c="r")
            r = r + 1
            # plt.draw()
            # plt.pause(0.0001)
        # plt.show()

        # Marking orphan clusters as bad clusters
        orphan_num = 1
        for c in range(0, nk):
            ioc = (ncluster == c).nonzero()
            ioc = ioc[0]
            if ioc.shape[0] <= orphan_num:
                # orphan sample... sorry
                bad_cluster[c] = 1

        # Eliminate bad clusters
        if (bad_cluster == 1).any():
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

        # Clustering Features
        cf_n = numpy.zeros((nk, 1))
        cf_ls = numpy.zeros((nk, n))
        cf_ss = numpy.zeros((nk, n))
        cf_time = numpy.zeros((nk, 1))
        cf_class = numpy.zeros((nk, 1))
        cf_weights = w
        cf_distance = dt
        cf_Si = Si
        # nit = nit

        for i in range(0, m):
            xi = X[i, :]
            # update Clustering Features
            ncls = int(ncluster[i])
            # test ncls of out of bounds indexes
            if ncls > (nk - 1):
                logging.debug('Hey! you were in a bad cluster %d', ncls)
            else:
                cf_n[ncls] = cf_n[ncls] + 1
                cf_ls[ncls, :] = cf_ls[ncls, :] + xi
                cf_ss[ncls, :] = cf_ss[ncls, :] + xi ** 2
                cf_class[ncls] = ccluster[ncls]

        # match new batch cluster class to model cluster class
        ds = numpy.zeros((nk, self.nk))
        dsv = numpy.zeros((nk, self.nk, n))
        ocluster = numpy.zeros((nk,))
        for i in range(0, nk):
            ci = fcluster[i, :]  # unlabeled batch cluster
            mind = -1
            for c in range(0, self.nk):
                xc = self.clusters[c, :]  # model cluster
                dd = distance_measure(xc, ci, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
                # dd = sqrt(sum((w. * (xc - xi)). ^ 2))
                ds[i, c] = dd
                dsv[i, c, :] = abs(xc - ci)
                if (mind == -1) | (dd < mind):
                    mind = dd
                    ocluster[i] = c
                    ccluster[i] = self.cf_class[c]

        # obtain classification
        output = -1 * numpy.ones((m, 1))
        nncs = numpy.argsort(ds, axis=1)
        # calculating radius
        Rc = numpy.zeros((self.nk,))
        Rcv = numpy.zeros((self.nk, self.n))
        for c in range(0, self.nk):
            Rc[c] = numpy.sqrt(
                sum(self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c]))
            Rcv[c] = numpy.sqrt(
                self.cf_weights[c, :] * (self.cf_ss[c, :] - (self.cf_ls[c, :] ** 2) / self.cf_n[c]) / self.cf_n[c])
        # incorporate new batch clusters to model
        for i in range(0, nk):
            Ri = numpy.sqrt(sum(cf_weights[i, :] * (cf_ss[i, :] - (cf_ls[i, :] ** 2) / cf_n[i]) / cf_n[i]))
            nnc = nncs[i, :]
            d1 = ds[i, nnc[0]]
            d2 = ds[i, nnc[1]]
            Rc1 = Rc[nnc[0]]
            Rc2 = Rc[nnc[1]]
            if cd == 0:  # cd1 batch cluster within model cluster radius
                if (d1 + Ri < Rc1) & (d2 + Ri > Rc2):
                    # if dic1 + ri < rc1 and dic2 + ri > rc2
                    c = nnc[0]
                    self.cf_n[c] = self.cf_n[c] + cf_n[i]
                    self.cf_ls[c, :] = self.cf_ls[c, :] + cf_ls[i, :]
                    self.cf_ss[c, :] = self.cf_ss[c, :] + cf_ss[i, :]
                    self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    output[ic] = self.cf_class[c]

                elif (d1 < Rc1) & (d2 > Rc2):
                    # if dic1 < rc1 and dic2 > rc2, relaxing
                    c = nnc[0]
                    self.cf_n[c] = self.cf_n[c] + cf_n[i]
                    self.cf_ls[c, :] = self.cf_ls[c, :] + cf_ls[i, :]
                    self.cf_ss[c, :] = self.cf_ss[c, :] + cf_ss[i, :]
                    self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    output[ic] = self.cf_class[c]
            elif cd == 1:  # cd2 minimum batch cluster to model cluster distance / radius ratio
                dr = ds[i, :] / Rc[:]
                ndrs = numpy.argsort(dr)
                c = ndrs[0]
                # if dic1 / rc1 is minimum
                self.cf_n[c] = self.cf_n[c] + cf_n[i]
                self.cf_ls[c, :] = self.cf_ls[c, :] + cf_ls[i, :]
                self.cf_ss[c, :] = self.cf_ss[c, :] + cf_ss[i, :]
                self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                ic = (ncluster == i).nonzero()
                ic = ic[0]
                output[ic] = self.cf_class[c]
            elif cd == 2:  # cd3 combination of cd1 and cd2
                dr = ds[i, :] / Rc[:]
                ndrs = numpy.argsort(dr)
                if (ndrs[0] == nnc[0]) & (d1 < Rc1):
                    # if min d1/Rc1 cluster = nearest cluster and distance to cluster is withing radius
                    c = ndrs[0]
                    self.cf_n[c] = self.cf_n[c] + cf_n[i]
                    self.cf_ls[c, :] = self.cf_ls[c, :] + cf_ls[i, :]
                    self.cf_ss[c, :] = self.cf_ss[c, :] + cf_ss[i, :]
                    self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    output[ic] = self.cf_class[c]
            elif cd == 3:  # cd3 includes cd1 but only samples within cluster radius are incorporated
                if ((d1 + Ri < Rc1) & (d2 + Ri > Rc2)) | ((d1 < Rc1) & (d2 > Rc2)):
                    # if dic1 + ri < rc1 and dic2 + ri > rc2
                    c = nnc[0]
                    d = nnc[1]
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        xc = self.clusters[c, :]
                        d1 = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
                        xd = self.clusters[d, :]
                        d2 = distance_measure(xd, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[d, :])
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < Rc1) & (d2 < Rc2):
                                nncluster = c
                            # se dentro do 3 x raio dos dois
                            elif (d1 < 3 * Rc1) & (d2 < 3 * Rc2):
                                nncluster = c
                            # se dentro do raio do mais próximo
                            elif (d1 < Rc1):
                                nncluster = c
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d2 > 2 * d1) & (d1 < Rc1) & (d2 > Rc2):
                                nncluster = c
                            # se distância do segundo é maior que 2 x a distância do primeiro e dentro do 3 x raio do primeiro e fora do raio do segundo
                            elif (d2 > 2 * d1) & (d1 < 3 * Rc1) & (d2 > Rc2):
                                nncluster = c
                            # se dentro do raio do mais próximo
                            elif (d1 < Rc1):
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]
            elif cd == 4:
                c = nnc[0]
                d = nnc[1]
                c1 = self.cf_class[c]
                c2 = self.cf_class[d]
                nncluster = -1
                if c1 == c2:
                    # se dentro do raio dos dois
                    if (d1 < Rc1) & (d2 < Rc2):
                        nncluster = c
                    # se dentro do 3 x raio dos dois
                    elif (d1 < 3 * Rc1) & (d2 < 3 * Rc2):
                        nncluster = c
                    # se dentro do raio do mais próximo
                    elif (d1 < Rc1):
                        nncluster = c
                    else:
                        nncluster = -1
                # se dois clusters tem classes diferentes
                else:
                    # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                    if (d2 > 2 * d1) & (d1 < Rc1) & (d2 > Rc2):
                        nncluster = c
                    # se distância do segundo é maior que 2 x a distância do primeiro e dentro do 3 x raio do primeiro e fora do raio do segundo
                    elif (d2 > 2 * d1) & (d1 < 3 * Rc1) & (d2 > Rc2):
                        nncluster = c
                    # se dentro do raio do mais próximo
                    elif (d1 < Rc1):
                        nncluster = c
                    else:
                        nncluster = -1

                if nncluster != -1:
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        xc = self.clusters[c, :]
                        d1 = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
                        xd = self.clusters[d, :]
                        d2 = distance_measure(xd, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[d, :])
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < Rc1) & (d2 < Rc2):
                                nncluster = c
                            # se dentro do 3 x raio dos dois
                            elif (d1 < 3 * Rc1) & (d2 < 3 * Rc2):
                                nncluster = c
                            # se dentro do raio do mais próximo
                            elif (d1 < Rc1):
                                nncluster = c
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d2 > 2 * d1) & (d1 < Rc1) & (d2 > Rc2):
                                nncluster = c
                            # se distância do segundo é maior que 2 x a distância do primeiro e dentro do 3 x raio do primeiro e fora do raio do segundo
                            elif (d2 > 2 * d1) & (d1 < 3 * Rc1) & (d2 > Rc2):
                                nncluster = c
                            # se dentro do raio do mais próximo
                            elif (d1 < Rc1):
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]
            elif cd == 5:  # cd4 simplified
                c = nnc[0]
                d = nnc[1]
                c1 = self.cf_class[c]
                c2 = self.cf_class[d]
                nncluster = -1
                if c1 == c2:
                    # se dentro do raio dos dois
                    if (d1 < Rc1) & (d2 < 3 * Rc2):
                        nncluster = c
                    else:
                        nncluster = -1
                # se dois clusters tem classes diferentes
                else:
                    # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                    if (d1 < 3 * Rc1) & (d2 > Rc2):
                        nncluster = c
                    else:
                        nncluster = -1

                if nncluster != -1:
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        # xc = self.clusters[c, :]
                        xc = (self.cf_n[c] * self.clusters[c, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[c] + cf_n[i])
                        d1 = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
                        # xd = self.clusters[d, :]
                        xd = (self.cf_n[d] * self.clusters[d, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[d] + cf_n[i])
                        d2 = distance_measure(xd, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[d, :])
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < (Rc1 + Ri)) & (d2 < 3 * (Rc2 + Ri)):
                                nncluster = c
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d1 < 3 * (Rc1 + Ri)) & (d2 > (Rc2 + Ri)):
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]
            elif cd == 6:  # same as cd5 but using vector distance
                c = nnc[0]
                d = nnc[1]
                c1 = self.cf_class[c]
                c2 = self.cf_class[d]
                nncluster = -1
                d1 = dsv[i, c, :]
                d2 = dsv[i, d, :]
                Rc1 = Rcv[c, :]
                Rc2 = Rcv[d, :]
                Ri = numpy.sqrt(cf_weights[i, :] * (cf_ss[i, :] - (cf_ls[i, :] ** 2) / cf_n[i]) / cf_n[i])
                if c1 == c2:
                    # se dentro do raio dos dois
                    if (d1 < 3 * Rc1).all() & (d2 < 3 * Rc2).all():
                        nncluster = c
                    else:
                        nncluster = -1
                # se dois clusters tem classes diferentes
                else:
                    # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                    if (d1 < 3 * Rc1).all() & (d2 > Rc2).any():
                        nncluster = c
                    else:
                        nncluster = -1

                if nncluster != -1:
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        xc = (self.cf_n[c] * self.clusters[c, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[c] + cf_n[i])
                        d1 = abs(xc - xi)
                        xd = (self.cf_n[d] * self.clusters[d, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[d] + cf_n[i])
                        d2 = abs(xd - xi)
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < 3 * (Rc1 + Ri)).all() & (d2 < 3 * (Rc2 + Ri)).all():
                                nncluster = c
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d1 < 3 * (Rc1 + Ri)).all() & (d2 > Rc2).any():
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]
            elif cd == 7:  # same as cd6 but using vector distance radius ratio
                c = nnc[0]
                d = nnc[1]
                c1 = self.cf_class[c]
                c2 = self.cf_class[d]
                nncluster = -1
                d1 = dsv[i, c, :]
                d2 = dsv[i, d, :]
                Rc1 = Rcv[c, :]
                Rc2 = Rcv[d, :]
                Ri = numpy.sqrt(cf_weights[i, :] * (cf_ss[i, :] - (cf_ls[i, :] ** 2) / cf_n[i]) / cf_n[i])
                if c1 == c2:
                    # se dentro do raio dos dois
                    if (d1 < 3 * Rc1).all() & (d2 < 3 * Rc2).all():
                        if (d1 / Rc1 < d2 / Rc2).all():
                            nncluster = c
                        elif (d1 / Rc1 > d2 / Rc2).all():
                            nncluster = d
                        else:
                            nncluster = -1
                    else:
                        nncluster = -1
                # se dois clusters tem classes diferentes
                else:
                    # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                    if (d1 < 3 * Rc1).all() & (d2 > Rc2).any():
                        nncluster = c
                    else:
                        nncluster = -1

                if nncluster != -1:
                    if nncluster == d:
                        d = c
                        c = nncluster
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        xc = self.clusters[c, :]
                        d1 = xc - xi
                        xd = self.clusters[d, :]
                        d2 = xd - xi
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < 3 * (Rc1 + Ri)).all() & (d2 < 3 * (Rc2 + Ri)).all():
                                if (d1 / (Rc1 + Ri) < d2 / (Rc2 + Ri)).all():
                                    nncluster = c
                                elif (d1 / (Rc1 + Ri) > d2 / (Rc2 + Ri)).all():
                                    nncluster = d
                                else:
                                    nncluster = -1
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d1 < 3 * (Rc1 + Ri)).all() & (d2 > Rc2).any():
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            c = nncluster
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]
            elif cd == 8:  # cd5 extended with distance radius ratio
                c = nnc[0]
                d = nnc[1]
                c1 = self.cf_class[c]
                c2 = self.cf_class[d]
                nncluster = -1
                if c1 == c2:
                    # se dentro do raio dos dois
                    if (d1 < Rc1) & (d2 < 3 * Rc2):
                        if d1 / Rc1 < d2 / Rc2:
                            nncluster = c
                        elif d1 / Rc1 > d2 / Rc2:
                            nncluster = d
                        else:
                            nncluster = -1
                    else:
                        nncluster = -1
                # se dois clusters tem classes diferentes
                else:
                    # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                    if (d1 < 3 * Rc1) & (d2 > Rc2):
                        nncluster = c
                    else:
                        nncluster = -1

                if nncluster != -1:
                    if nncluster == d:
                        d = c
                        c = nncluster
                    ic = (ncluster == i).nonzero()
                    ic = ic[0]
                    for j in ic:
                        # for each sample of batch cluster i, get distance to 2 nearest clusters
                        xi = X[j, :]
                        # xc = self.clusters[c, :]
                        xc = (self.cf_n[c] * self.clusters[c, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[c] + cf_n[i])
                        d1 = distance_measure(xc, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[c, :])
                        # xd = self.clusters[d, :]
                        xd = (self.cf_n[d] * self.clusters[d, :] + cf_n[i] * fcluster[i, :]) / (self.cf_n[d] + cf_n[i])
                        d2 = distance_measure(xd, xi, t=self.cf_distance, Si=self.cf_Si, w=self.cf_weights[d, :])
                        # se dois clusters tem classes iguais
                        c1 = self.cf_class[c]
                        c2 = self.cf_class[d]
                        nncluster = -1
                        if c1 == c2:
                            # se dentro do raio dos dois
                            if (d1 < (Rc1 + Ri)) & (d2 < 3 * (Rc2 + Ri)):
                                if d1 / (Rc1 + Ri) < d2 / (Rc2 + Ri):
                                    nncluster = c
                                elif d1 / (Rc1 + Ri) > d2 / (Rc2 + Ri):
                                    nncluster = d
                                else:
                                    nncluster = -1
                            else:
                                nncluster = -1
                        # se dois clusters tem classes diferentes
                        else:
                            # se distância do segundo é maior que 2 x distância do primeiro e dentro do raio do primeiro e fora do raio do segundo
                            if (d1 < 3 * (Rc1 + Ri)) & (d2 > (Rc2 + Ri)):
                                nncluster = c
                            else:
                                nncluster = -1

                        if nncluster != -1:
                            c = nncluster
                            self.cf_n[c] = self.cf_n[c] + 1
                            self.cf_ls[c, :] = self.cf_ls[c, :] + xi
                            self.cf_ss[c, :] = self.cf_ss[c, :] + xi ** 2
                            self.clusters[c, :] = self.cf_ls[c, :] / self.cf_n[c]
                            output[ic] = self.cf_class[c]

        # plot results
        # plt.scatter(X[:, 0], X[:, 1], marker="o", c=T)
        # plt.scatter(X[:, 0], X[:, 1], marker="+", c=output[:, 0])
        # plt.show()

        return output


class ISSKDEKMeansEnsemble(BaseEstimator):
    def __init__(self, nk=2, n=2, w=0, dt='euclidean', cini=True, nit=100, thd=0, alpha=0.75, mo=1, ver=2):
        self.nk = nk
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
        self.mode = 1  # 0: vote; 1: distance
        self.offline_training_time = 0.0

    def fit(self, X, y):
        _start_time = time.time()
        self.n = X.shape[1]
        self.nkm = self.n + 1
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
            self.kms.append(
                ISSKDEKMeans(self.nk, sum(self.fsel[i, :]), self.w[:, self.fsel[i, :]], self.dt, self.cini, self.nit,
                             self.thd, self.alpha, self.mo, self.ver))
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
