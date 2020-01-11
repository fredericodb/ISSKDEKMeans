import numpy
import matplotlib.pyplot as plt
import math

import logging


class IELM:

    def __init__(self, nHiddenNeurons=180, ActivationFunction='sig', Block=20):
        # 0, 25, 'rbf', 75, 1
        # 1, 180, 'sig', 280, 20
        self.nHiddenNeurons = nHiddenNeurons
        self.ActivationFunction = ActivationFunction
        self.Block = Block

    def RBFun(self, P, IW, Bias):

        ######## RBF network using Gaussian kernel
        V = P @ IW.T
        ind = numpy.zeros((P.shape[0], 1), dtype=numpy.int64)
        for i in range(0, IW.shape[0]):
            Weight = IW[i, :]
            WeightMatrix = Weight[ind, :]
            V[:, i] = -numpy.sum((P - WeightMatrix) ** 2, axis=1)

        BiasMatrix = Bias[ind, :]
        V = V * BiasMatrix
        H = numpy.exp(V)
        H = H[0]

        return H

    def HardlimActFun(self, P, IW, Bias):

        ######## Feedforward neural network using hardlim activation function
        V = P @ IW.T
        ind = numpy.zeros((1, P.shape[0]), dtype=numpy.int64)
        BiasMatrix = Bias[ind, :]
        V = V + BiasMatrix
        H = numpy.array(V > 0.0, dtype=float)  # hardlim
        H = H[0]

        return H

    def SigActFun(self, P, IW, Bias):

        ######## Feedforward neural network using sigmoidal activation function
        V = P @ IW.T
        ind = numpy.zeros((1, P.shape[0]), dtype=numpy.int64)
        BiasMatrix = Bias[ind, :]
        V = V + BiasMatrix
        H = 1. / (1 + numpy.exp(-V))
        H = H[0]

        return H

    def SinActFun(self, P, IW, Bias):

        ######## Feedforward neural network using sine activation function
        V = P @ IW.T
        ind = numpy.zeros((1, P.shape[0]), dtype=numpy.int64)
        BiasMatrix = Bias[ind, :]
        V = V + BiasMatrix
        H = numpy.sin(V)
        H = H[0]

        return H

    def fit(self, P, T):
        N0 = P.shape[0]
        nInputNeurons = P.shape[1]
        a = T
        T = numpy.zeros((N0, a.max()+1))
        T[numpy.arange(N0), a.astype(dtype=numpy.int64)] = 1
        self.nInputNeurons = nInputNeurons
        self.nOutputNeurons = a.max()+1

        # start_time_train = cputime
        ########### step 1 Initialization Phase
        P0 = P[0:N0, :]
        T0 = T[0:N0, :]

        self.IW = numpy.random.random_sample((self.nHiddenNeurons, nInputNeurons)) * 2 - 1
        if self.ActivationFunction.lower() == 'rbf':
            self.Bias = numpy.random.random_sample((1, self.nHiddenNeurons))
            # Bias = rand(1,nHiddenNeurons)*1/3+1/11     ############# for the cases of Image Segment and Satellite Image
            # Bias = rand(1,nHiddenNeurons)*1/20+1/60    ############# for the case of DNA
            H0 = self.RBFun(P0, self.IW, self.Bias)
        elif self.ActivationFunction.lower() == 'sig':
            self.Bias = numpy.random.random_sample((1, self.nHiddenNeurons)) * 2 - 1
            H0 = self.SigActFun(P0, self.IW, self.Bias)
        elif self.ActivationFunction.lower() == 'sin':
            self.Bias = numpy.random.random_sample((1, self.nHiddenNeurons)) * 2 - 1
            H0 = self.SinActFun(P0, self.IW, self.Bias)
        elif self.ActivationFunction.lower() == 'hardlim':
            self.Bias = numpy.random.random_sample((1, self.nHiddenNeurons)) * 2 - 1
            H0 = self.HardlimActFun(P0, self.IW, self.Bias)
            H0 = H0.astype(dtype=numpy.double)

        self.M = numpy.linalg.pinv(H0.T @ H0)
        self.beta = numpy.linalg.pinv(H0) @ T0

    def partial_fit(self, P, T):
        nTrainingData = P.shape[0]
        nInputNeurons = P.shape[1]
        a = T
        T = numpy.zeros((nTrainingData, self.nOutputNeurons))
        T[numpy.arange(nTrainingData), a.astype(dtype=numpy.int64)] = 1

        ############# step 2 Sequential Learning Phase
        for n in range(0, nTrainingData, self.Block):
            if (n + self.Block) > nTrainingData:
                Pn = P[n:nTrainingData, :]
                Tn = T[n:nTrainingData, :]
                self.Block = Pn.shape[0]  #### correct the block size
                # clear V  #### correct the first dimention of V
            else:
                Pn = P[n:(n + self.Block), :]
                Tn = T[n:(n + self.Block), :]

            if self.ActivationFunction.lower() == 'rbf':
                H = self.RBFun(Pn, self.IW, self.Bias)
            elif self.ActivationFunction.lower() == 'sig':
                H = self.SigActFun(Pn, self.IW, self.Bias)
            elif self.ActivationFunction.lower() == 'sin':
                H = self.SinActFun(Pn, self.IW, self.Bias)
            elif self.ActivationFunction.lower() == 'hardlim':
                H = self.HardlimActFun(Pn, self.IW, self.Bias)

            self.M = self.M - self.M @ H.T @ numpy.linalg.inv(numpy.eye(self.Block) + H @ self.M @ H.T) @ H @ self.M
            self.beta = self.beta + self.M @ H.T @ (Tn - H @ self.beta)

        # _time_train = cputime
        # TrainingTime = _time_train - start_time_train

    def predict(self, P):
        ########### Performance Evaluation
        # start_time_test = cputime
        if self.ActivationFunction.lower() == 'rbf':
            HTest = self.RBFun(P, self.IW, self.Bias)
        elif self.ActivationFunction.lower() == 'sig':
            HTest = self.SigActFun(P, self.IW, self.Bias)
        elif self.ActivationFunction.lower() == 'sin':
            HTest = self.SinActFun(P, self.IW, self.Bias)
        if self.ActivationFunction.lower() == 'hardlim':
            HTest = self.HardlimActFun(P, self.IW, self.Bias)

        TY = HTest @ self.beta

        TY = numpy.argmax(TY, axis=1)

        # _time_test = cputime
        # TestingTime = _time_test - start_time_test

        return TY
