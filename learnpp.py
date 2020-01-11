import numpy
import matplotlib.pyplot as plt
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import logging

class LearnPP:

    def __init__(self, base_classifier='CART', Tk=3):
        self.base_classifier = base_classifier
        self.classifiers = [] # cell(Tk*K, 1)  # cell array with total number of classifiers
        self.beta = [] # numpy.zeros((Tk*K, 1)) # beta will set the classifier weights
        self.mclass = 0
        self.c_count = 0 # keep track of the number of classifiers at each time
        self.Tk = Tk # number of classifiers to generate
        self.errs = [] # numpy.zeros((Tk*K, 1))   # prediction errors on the test data set

    def count(self):
        return self.c_count

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### AUXILARY FUNCTIONS
    def classify_ensemble(self, data, n_experts=-1):
        if n_experts == -1:
            n_experts = self.c_count
        if 0 in self.beta[0:n_experts]:
            weights = numpy.inf * numpy.array(numpy.arange(0, n_experts))
        else:
            weights = numpy.log(1 / numpy.array(self.beta[0:n_experts]))
        p = numpy.zeros((data.shape[0], self.mclass))
        for k in range(0, n_experts):
            y = self.classifiers[k].predict(data)

            # this is inefficient, but it does the job
            for m in range(0, len(y)):
                p[m, y[m]] = p[m, y[m]] + weights[k]

        predictions = numpy.argmax(p, axis=1)
        posterior = p / numpy.tile(numpy.sum(p, axis=1), (self.mclass, 1)).T

        return predictions, posterior


    def fit(self, X, T):
        #   [net,errs] = learn(net, data_train, labels_train, ...
        #     data_test, labels_test)
        #
        #     @net - initialized structure. you must initialize
        #       net.iterations
        #       net.base_classifier - you should set this to be model.type
        #         which is submitted to CLASSIFIER_TRAIN.m
        #       net.mclass - number of classes
        #     @data_train - training data in a cell array. each entry should
        #         have a n_oberservation by n_feature matrix
        #     @labels_train - cell array of class labels
        #     @data_test - test data in a matrix. the size of the matrix should
        #         be n_oberservation by n_feature matrix
        #     @labels_test -labels to the test data
        #     @errs - error of the Learn++ on the testing data set. error is
        #       measured at each addition of a new classifier.
        #
        #   Implementation of Learn++.
        #
        #   Cite:
        #   1) R. Polikar, L. Udpa, S. Udpa, and V. Honavar, "Learn++: An
        #      incremental learning algorithm for supervised neural networks,"
        #      IEEE Transactions on System, Man and Cybernetics (C), Special
        #      Issue on Knowledge Management, vol. 31, no. 4, pp. 497-508, 2001.
        #
        #   See also
        #   CLASSIFIER_TRAIN.m CLASSIFIER_TEST.m

        #     learn.m
        #     Copyright (C) 2013 Gregory Ditzler
        #
        #     This program is free software: you can redistribute it and/or modify
        #     it under the terms of the GNU General Public License as published by
        #     the Free Software Foundation, either version 3 of the License, or
        #     (at your option) any later version.
        #
        #     This program is distributed in the hope that it will be useful,
        #     but WITHOUT ANY WARRANTY without even the implied warranty of
        #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        #     GNU General Public License for more details.
        #
        #     You should have received a copy of the GNU General Public License
        #     along with this program.  If not, see <http://www.gnu.org/licenses/>.

        # obtain the latest data set and initialize the weights over the
        # instances to form a uniform distribution
        data_train_k = X  # FOR K == 1 THE DATA SET MUST BE THE LABELED DATA
        labels_train_k = T  # RUN SRS HERE ON UNLABELED DATA ONLY

        if len(numpy.unique(labels_train_k)) > self.mclass:
            self.mclass = len(numpy.unique(labels_train_k))

        D = numpy.ones((len(labels_train_k), 1)) / len(labels_train_k)

        # original paper says to modify D if prior knowledge is available. we can
        # modify the distribution weights if we already have a classifier
        # ensemble.
        isunlabeled = labels_train_k[0] == -1

        if isunlabeled:  # ALL OTHER DATA SETS CONSIST OF UNLABELED DATA
            labels_train_k = -1 # assign label to unlabeled data - SRS_multiclasses(data_train[1], labels_train[1], data_train_k, net.base_classifier.type)

        if self.c_count > 0:
            predictions, _ = self.classify_ensemble(data_train_k, self.c_count)  # predict on the training data
            epsilon_kt = sum(D[predictions != labels_train_k])  # error on D
            if epsilon_kt == 0:
                epsilon_kt = epsilon_kt + 1e-10  # minha DO CHECK epsilon_kt BEFORE ASSIGNING
                return

            beta_kt = epsilon_kt / (1 - epsilon_kt)  # normalized error on D
            D[predictions == labels_train_k] = beta_kt * D[predictions == labels_train_k]

        for t in range(0, self.Tk):
            # step 1 - make sure we are working with a probability distribution.
            D = D / sum(D)

            # step 2 - grab a random sample of data indices with replacement from
            # the probability distribution D
            index = numpy.random.choice(numpy.arange(0,len(D)), len(D), True, numpy.reshape(D,len(D)))

            # step 3 - generate a new classifier on the data sampled from D.
            if self.base_classifier == 'CART':
                self.classifiers.append(DecisionTreeClassifier())
            elif self.base_classifier == 'NN':
                n_features = data_train_k.shape[1]
                n_hidden_neurons = 2 * n_features + 1
                self.classifiers.append(MLPClassifier(hidden_layer_sizes=(n_hidden_neurons,)))

            self.classifiers[self.c_count].fit(data_train_k[index, :], labels_train_k[index])

            # step 4 - test the latest classifier on ALL of the data not just the
            # data sampled from D, and compute the error according to the
            # probability distribution. then compute beta
            y = self.classifiers[self.c_count].predict(data_train_k)
            epsilon_kt = sum(D[y != labels_train_k])
            self.beta.append(epsilon_kt / (1 - epsilon_kt))

            # update the classifier count
            self.c_count = self.c_count + 1

            # step 5 - get the ensemble decision computed with c_count classifiers
            # in the ensemble. compute the error on the probability distribution on
            # the composite hypothesis.
            predictions, _ = self.classify_ensemble(data_train_k, self.c_count)
            E_kt = sum(D[predictions != labels_train_k])
            if E_kt > 0.5:
                # rather than remove remove existing classifier null the result out
                # by forcing the loss to be equal to 1/2 which is the worst possible
                # loss. feel free to modify the code to go back an iteration.
                E_kt = 0.5

            # step 6 - compute the normalized error of the compsite hypothesis and
            # update the weights over the training instances in the kth batch.
            if E_kt == 0:
                E_kt = E_kt + 1e-10  # minha DO CHECK E_kt BEFORE ASSIGNING

            Bkt = E_kt / (1 - E_kt)
            D[predictions == labels_train_k] = Bkt * D[predictions == labels_train_k]
            D = D / sum(D)

            # make some predictions on the testing data set.
            #predictions, posterior = self.classify_ensemble(data_test, self.c_count)
            #self.errs[self.c_count] = sum(numpy.nonzero(predictions != labels_test)) / len(labels_test)

    def predict(self, X):

        output, _ = self.classify_ensemble(X)
        return output

    def partial_fit(self, xi, yi):

        self.fit(xi, yi)
        return

    def fit_unlabeled(self, xi):

        return

    def fit_unlabeled_batch(self, X):

        return
