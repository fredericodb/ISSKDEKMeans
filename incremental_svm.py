import numpy
import matplotlib.pyplot as plt
import math
import random

import logging

class IncrementalSVM:

    def __init__(self):
        # define global variables
        self.a = 0  # alpha coefficients
        self.b = 0  # bias
        self.C = 0  # regularization parameters
        self.deps = 0  # jitter factor in kernel matrix
        self.g = 0  # partial derivatives of cost function w.r.t. alpha coefficients
        self.ind = {}  # cell array containing indices of margin, error, reserve and unlearned vectors
        self.kernel_evals = 0  # kernel evaluations
        self.max_reserve_vectors = 0  # maximum number of reserve vectors stored
        self.perturbations = 0  # number of perturbations
        self.Q = 0  # extended kernel matrix for all vectors
        self.Rs = 0  # inverse of extended kernel matrix for margin vectors
        self.scale = 0  # kernel scale
        self.type = 0  # kernel type
        self.uind = 0  # user-defined example indices
        self.X = 0  # matrix of margin, error, reserve and unlearned vectors stored columnwise
        self.y = 0  # column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors
        self.labels_ = 0
        self.accuracy = 0.0

    # KERNEL - Kernel function evaluation
    #
    # Syntax: K = kernel(X,Y,type,scale)
    #
    #      X: (N,Mx) dimensional matrix
    #      Y: (N,My) dimensional matrix
    #      K: (Mx,My) dimensional matrix 
    #   type: kernel type
    #           1: linear kernel        X'*Y
    #         2-4: polynomial kernel    (scale*X'*Y + 1)^type
    #           5: Gaussian kernel with variance 1/(2*scale)
    #  scale: kernel scale
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def kernel(self, X, Y, type, scale):

        K = X.T @ Y
        if len(X.shape) == 1:
            X = numpy.reshape(X, (X.size, 1))
        (N, Mx) = X.shape
        if len(Y.shape) == 1:
            Y = numpy.reshape(Y, (1, Y.size))
        (N, My) = Y.shape
        if (type > 1) and (type < 5):
            K = (K * scale + 1) ** type
        elif type == 5:
            K = 2 * K
            K = K - numpy.sum(X**2, axis=0).T * numpy.ones((1, My)) # @
            K = K - numpy.ones((Mx, 1)) * numpy.sum(Y**2, axis=0) # @
            K = numpy.exp(K / (2 * scale))



        self.kernel_evals = self.kernel_evals + Mx * My
        
        return K

    # MOVE_IND - Removes the indices indc from the indices inda
    #            and appends them to end of the indices indb.
    #            The relative order of the remaining indices in
    #            inda is preserved.
    #
    # Syntax: [inda,indb] = move_ind(inda,indb,indc)
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def move_ind(self, inda, indb, indc):

        if not numpy.all(indc == 0):
            indb = numpy.hstack((indb, indc))
            new_inds = []
            for i in range(0,len(inda)):
                if not numpy.any(inda[i] == indc):
                    new_inds = numpy.hstack((new_inds, inda[i]))

            inda = new_inds

        return inda,indb


    # MOVE_INDR - Removes the indices indc from the indices inda
    #             and adds them to the reserve vector index list
    #             if necessary.
    #
    # Syntax: [inda,i] = move_indr(inda,indc)
    #
    #      i: when i > 0, the i-th element was removed from ind{RESERVE}
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def move_indr(self, inda, indc):

        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        # define global variables
        g = self.g  # partial derivatives of cost function w.r.t. alpha coefficients
        ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors
        max_reserve_vectors = self.max_reserve_vectors  # maximum number of reserve vectors stored

        removed_i = 0
        num_RVs_orig = len(ind[RESERVE])

        # shift indc from inda to ind[RESERVE]
        inda, ind[RESERVE] = self.move_ind(inda, ind[RESERVE], indc)

        # if we need to remove some reserve vectors
        if len(ind[RESERVE]) > max_reserve_vectors:

            # sort g(ind[RESERVE])
            i = g[ind[RESERVE]].argsort()
            g_sorted = g[ind[RESERVE]].sort()

            # reserve vectors that need to be removed
            removed = i[max_reserve_vectors + 1:len(i)]

            # find any original reserve vectors that need to be removed
            k = numpy.nonzero(removed <= num_RVs_orig)
            if len(k) > 0:
                removed_i = removed[k]

            # remove the necessary reserve vectors
            ind[RESERVE][removed] = []

        return inda, removed_i

    # BOOKKEEPING - Updates the status of example indss and modifies the corresponding
    #               coefficient a(indss) in certain cases to avoid numerical errors.
    #
    # Syntax: [indco,i] = bookkeeping(indss,cstatus,nstatus)
    #
    #     indco: matrix row/col to remove from Rs and Q if removing a margin vector
    #         i: when i > 0, the i-th element was removed from ind{RESERVE}
    #     indss: the example changing status
    #   cstatus: current status of the example
    #   nstatus: new status of the example
    #
    #            example status values:
    #            1: margin vector
    #            2: error vector
    #            3: reserve vector
    #            4: unlearned vector
    #
    # Version 3.22 -- Comments to diehl@alumni.cmu.edu
    #

    def bookkeeping(self, indss, cstatus, nstatus):

        indco = -1
        i = 0
        if cstatus != nstatus:

            # flags for example state
            MARGIN = 0
            ERROR = 1
            RESERVE = 2
            UNLEARNED = 3

            # define global variables
            a = self.a  # the alpha coefficients
            C = self.C  # regularization parameters
            ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors

            # adjust coefficient to avoid numerical errors if necessary
            if nstatus == RESERVE:
                a[indss] = 0
            elif nstatus == ERROR:
                a[indss] = C[indss]

            # if the example is currently a margin vector, determine the row
            # in the extended kernel matrix inverse that needs to be removed
            if cstatus == MARGIN:
                indco = numpy.nonzero(indss == ind[MARGIN]) + 1


            # change the status of the example
            if nstatus == RESERVE:
                ind[cstatus], i = self.move_indr(ind[cstatus], indss)
            else:
                ind[cstatus], ind[nstatus] = self.move_ind(ind[cstatus], ind[nstatus], indss)

            self.a = a
            self.ind = ind

        return indco, i

    # UPDATERQ - Updates Rs and Q accordingly when adding or removing a
    #            margin vector.  Note: Rs and Q are defined as global
    #            variables.
    #
    # Syntax: updateRQ(beta,gamma,indc)
    #         (for adding a margin vector)
    #
    #         updateRQ(indc)
    #         (for removing a margin vector)
    #
    #   beta: parameter sensitivities associated with the example indc
    #  gamma: margin sensitivity associated with the example indc
    #   indc: example/matrix index
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def updateRQ(self, beta=None, gamma=None, indc=None):

        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        # define global variables
        C = self.C  # regularization parameters
        deps = self.deps  # jitter factor in kernel matrix
        ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors
        Q = self.Q  # extended kernel matrix for all vectors
        Rs = self.Rs  # inverse of extended kernel matrix for margin vectors
        scale = self.scale  # kernel scale
        type = self.type  # kernel type
        X = self.X  # matrix of margin, error, reserve and unlearned vectors stored columnwise
        y = self.y  # column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors


        if beta != None:
            expand = 1
        elif indc != None:
            expand = 0
        else:
            print('updateRQ: Incorrect number of parameters')

        rows = Rs.shape[0]
        if expand:

            if gamma < deps:
                gamma = deps

            if rows > 1:
                Rs = numpy.array([[Rs, numpy.zeros((rows, 1))], [numpy.zeros((1, rows + 1))]]) + numpy.array([[beta], [1]]) @ numpy.array([beta.T, 1])/gamma
            else:
                Rs = numpy.array([[-(self.kernel(X[:, indc], X[:, indc], type, scale) + deps), y[indc]], [y[indc], 0]])

            Q = numpy.array([[Q], [(y[indc] @ y.T) * self.kernel(X[:,indc],X,type,scale)]])
            Q[rows + 1, indc] = Q[rows + 1, indc] + deps

        else:

            if rows > 2:
                stripped = numpy.array([numpy.arange(0, (indc-1)), numpy.arange((indc+1), Rs.shape[0])])
                Rs = Rs[stripped, stripped] - Rs[stripped, indc] @ Rs[indc, stripped] / Rs[indc, indc]
            else:
                Rs = numpy.inf

            Q[indc, :] = []

        self.Rs = Rs
        self.Q = Q

    # MIN_DELTA - Computes the minimum change in the parameter lambda
    #             that causes one of the parameters psi(k) to change from
    #             the given initial value to the given final value.
    #             Only the parameters with flags(k) = 1 are checked.
    #             psi(k) and lambda are assumed to be linearly related.
    #
    # Syntax: [min_d,k] = min_delta(flags,psi_initial,psi_final,psi_sens)
    #
    #       flags: parameters to check
    #       min_d: minimum change in lambda
    #           k: parameter psi_k that achieves psi_final_k
    # psi_initial: initial values for the parameters psi_i
    #   psi_final: final values for the parameters psi_i
    #    psi_sens: parameter sensitivities (d(psi_i)/d(lambda))
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def min_delta(self, flags, psi_initial, psi_final, psi_sens):

        if numpy.any(flags):

            # find the parameters to check
            ind = numpy.nonzero(flags)

            deltas = (psi_final[ind] - psi_initial[ind]) / psi_sens[ind]
            min_d, i = numpy.min(deltas)
            k = ind[i]

            # if there is more than one parameter that achieves the final value
            # with min_delta, select the parameter with the highest
            # sensitivity |psi_sens|
            min_k = numpy.nonzero(deltas == min_d)
            if len(min_k) > 1:
                i = numpy.argmax(abs(psi_sens[ind[min_k]]))
                max_sens = numpy.max(abs(psi_sens[ind[min_k]]))
                k = ind[min_k[i]]

        else:

            min_d = numpy.inf
            k = -1

        return min_d, k

    # MIN_DELTA_ACB - Computes the minimum acceptable change in alpha_c or b
    #                 and indicates which example changes status along
    #                 with the type of status change.
    #
    # Syntax: [min_dacb,indss,cstatus,nstatus] = min_delta_acb(indc,gamma,beta,polc,rflag)
    #
    #   min_dacb: minimum acceptable change in alpha_c (num MVs > 0) / b (num MVs = 0)
    #      indss: the example changing status
    #    cstatus: current status of the example
    #             0: if indss = indc
    #             1: margin vector
    #             2: error vector
    #             3: reserve vector
    #             4: unlearned vector
    #    nstatus: new status of the example
    #             returns one of the values above
    #       indc: the current example being learned/unlearned
    #      gamma: margin sensitivities (delta g / delta alpha_c/b)
    #       beta: coefficient sensitivities (delta alpha_k/b / delta alpha_c/b)
    #       polc: sign of the change in alpha_c/b (+1/-1)
    #      rflag: flag indicating whether or not to check if any reserve vectors
    #             become margin vectors during learning
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def min_delta_acb(self, indc, gamma, beta, polc, rflag):

        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        # define global variables
        a = self.a  # the alpha coefficients
        C = self.C  # regularization parameter(s)
        g = self.g  # partial derivatives of the objective function w.r.t. the alphas
        ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors

        indss = numpy.zeros((6, 1))
        indss[0:3] = indc
        cstatus = numpy.zeros((6, 1))
        nstatus = numpy.zeros((6, 1))

        # upper limits on change in alpha_c or b assuming no other examples change status
        if polc == 1:
            delta_m = -g[indc] / gamma[indc]
            nstatus[0] = MARGIN
            if beta.size > 1:  # if there are margin vectors
                delta_e = C[indc] - a[indc]
                nstatus[1] = ERROR
            else:  # only the bias term is allowed to change so indc can't be an error vector
                delta_e = numpy.inf

            delta_r = numpy.inf
        else:
            if g[indc] > 0:  # decrementing indc when a(indc) > 0 and g(indc) > 0 during kernel perturbation
                delta_m = g[indc] / gamma[indc]
                nstatus[0] = MARGIN
            else:
                delta_m = numpy.inf

            if a[indc] <= C[indc]:
                delta_e = numpy.inf
                delta_r = a[indc]
                if g[indc] > 0:  # decrementing indc when a(indc) > 0 and g(indc) > 0 during kernel perturbation
                    nstatus[2] = RESERVE
                else:
                    nstatus[2] = UNLEARNED

            else:  # decrementing indc when a(indc) > C(indc) during reg. parameter perturbation
                delta_e = a[indc] - C[indc]
                delta_r = numpy.inf
                nstatus[1] = ERROR

        # change in alpha_c or b that causes a margin vector to change to an error vector
        # or reserve vector
        if beta.size > 1:  # if there are margin vectors
            beta_s = polc * beta[1:len(beta)]
            flags = (abs(beta_s) > 0)
            delta_mer, i = self.min_delta(flags, a[ind[MARGIN]], C[ind[MARGIN]] * (beta_s > 0), beta_s)
            if delta_mer < numpy.inf:
                indss[3] = ind[MARGIN][i]
                cstatus[3] = MARGIN
                nstatus[3] = ERROR * (beta_s[i] > 0) + RESERVE * (beta_s[i] < 0)

        else:
            delta_mer = numpy.inf

        # change in alpha_c or b that causes an error vector to change to a margin vector
        gamma_e = polc * gamma[ind[ERROR]]
        flags = (gamma_e > 0)
        delta_em, i = self.min_delta(flags, g[ind[ERROR]], numpy.zeros((len(ind[ERROR]), 1)), gamma_e)
        if delta_em < numpy.inf:
            indss[4] = ind[ERROR][i]
            cstatus[4] = ERROR
            nstatus[4] = MARGIN

        # change in alpha_c or b that causes a reserve vector to change to a margin vector
        if rflag:
            gamma_r = polc * gamma[ind[RESERVE]]
            flags = numpy.logical_and(g[ind[RESERVE]] >= 0, gamma_r < 0)
            delta_rm, i = self.min_delta(flags, g[ind[RESERVE]], numpy.zeros((len(ind[RESERVE]), 1)), gamma_r)
            if delta_rm < numpy.inf:
                indss[5] = ind[RESERVE][i]
                cstatus[5] = RESERVE
                nstatus[5] = MARGIN

        else:
            delta_rm = numpy.inf

            # minimum acceptable value for |delta_ac| or |delta_b|
        min_ind = numpy.argmin(numpy.array([delta_m, delta_e, delta_r, delta_mer, delta_em, delta_rm]))
        min_dacb = numpy.min(numpy.array([delta_m, delta_e, delta_r, delta_mer, delta_em, delta_rm]))
        indss = indss[min_ind]
        cstatus = cstatus[min_ind]
        nstatus = nstatus[min_ind]

        # multiply by the proper sign to yield delta_ac or delta_b
        min_dacb = polc * min_dacb

        return min_dacb, indss, cstatus, nstatus


    # SVMEVAL - Evaluates a support vector machine at the given data points.
    #
    # Syntax: [f,K] = svmeval(X,a,b,ind,X_mer,y_mer,type,scale)
    #         (evaluates the given SVM at the data points contained in X)
    #
    #         [f,K] = svmeval(X)
    #         (evaluates the SVM in memory at the data points contained in X)
    #
    #      f: SVM output for the evaluation vectors
    #      K: kernel matrix containing dot products in feature space between
    #         the margin and error vectors (rows of K) and the column vectors in X
    #         (columns of K)
    #      X: matrix of evaluation vectors stored columnwise
    #      a: alpha coefficients
    #      b: bias
    #    ind: cell array containing indices of margin, error and reserve vectors
    #         ind{1}: indices of margin vectors
    #         ind{2}: indices of error vectors
    #         ind{3}: indices of reserve vectors
    #  X_mer: matrix of margin, error and reserve vectors stored columnwise
    #  y_mer: column vector of class labels (-1/+1) for margin, error and reserve vectors
    #   type: kernel type
    #           1: linear kernel        K(x,y) = x'*y
    #         2-4: polynomial kernel    K(x,y) = (scale*x'*y + 1)^type
    #           5: Gaussian kernel with variance 1/(2*scale)
    #  scale: kernel scale
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def svmeval(self, X_eval, a=None, b=None, ind=None, X=None, y=None, type=None, scale=None):

        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        if a == None:
            a = self.a
            b = self.b
            ind = self.ind
            X = self.X
            y = self.y
            type = self.type
            scale = self.scale

        # evaluate the SVM

        # find all of the nonzero coefficients
        # (note: when performing kernel perturbation, ind{MARGIN} and ind{ERROR}
        #  do not necessarily identify all of the nonzero coefficients)
        indu = []
        if UNLEARNED in ind:
            u = a[ind[UNLEARNED]] > 0
            if u.any():
                indu = numpy.nonzero(u)
                indu = ind[UNLEARNED][indu]
        indr = []
        if RESERVE in ind:
            r = a[ind[RESERVE]] > 0
            if r.any():
                indr = numpy.nonzero(r)
                indr = ind[RESERVE][indr]
        indme = []
        if MARGIN in ind and ERROR in ind:
            indme = numpy.array([ind[MARGIN], ind[ERROR]])

        K = []
        f = b
        if len(indme) > 0:
            K = self.kernel(X[:, indme], X_eval, type, scale)
            f = f + K.T @ (y[indme] * a[indme])

        if len(indu) > 0:
            f = f + self.kernel(X[:, indu], X_eval, type, scale).T @ (y[indu] * a[indu])

        if len(indr) > 0:
            f = f + self.kernel(X[:, indr], X_eval, type, scale).T @ (y[indr] * a[indr])

        return f, K


    # LEARN - Increments the specified example into the current SVM solution.
    #         Assumes alpha_c = 0 initially.
    #
    # Syntax: nstatus = learn(indc,rflag)
    #
    # nstatus: new status for indc
    #    indc: index of the example to learn
    #   rflag: flag indicating whether or not to check if any reserve vectors
    #          become margin vectors during learning
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def learn(self, indc, rflag):

        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        # define global variables
        a = self.a  # alpha coefficients
        b = self.b  # bias
        C = self.C  # regularization parameters
        deps = self.deps  # jitter factor in kernel matrix
        g = self.g  # partial derivatives of cost function w.r.t. alpha coefficients
        ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors
        perturbations = self.perturbations  # number of perturbations
        Q = self.Q  # extended kernel matrix for all vectors
        Rs = self.Rs  # inverse of extended kernel matrix for margin vectors
        scale = self.scale  # kernel scale
        type = self.type  # kernel type
        X = self.X  # matrix of margin, error, reserve and unlearned vectors stored columnwise
        y = self.y  # column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors

        # compute g(indc)
        f_c, K = self.svmeval(X[:, indc])
        g[indc] = y[indc] * f_c - 1

        # if g(indc) > 0, place this example into the reserve set directly
        if g[indc] >= 0:
            # move the example to the reserve set
            self.bookkeeping(indc, UNLEARNED, RESERVE)
            nstatus = RESERVE

            return

        # compute Qcc and Qc if necessary
        if MARGIN not in ind:
            num_MVs = 0
        else:
            num_MVs = len(ind[MARGIN])
        Qc = {}
        if num_MVs == 0:
            if ERROR in ind:
                if len(ind[ERROR]) > 0:
                    Qc[ERROR] = (y[ind[ERROR]] @ y[indc]) * self.kernel(X[:, ind[ERROR]], X[:, indc], type, scale)

        else:
            if MARGIN in ind:
                Qc[MARGIN] = (y[ind[MARGIN]] @ y[indc]) * K[0:num_MVs]
                if ERROR in ind:
                    if len(ind[ERROR]) > 0:
                        Qc[ERROR] = (y[ind[ERROR]] @ y[indc]) * K[num_MVs:len[K]]

        if RESERVE in ind:
            if len(ind[RESERVE]) > 0:
                Qc[RESERVE] = (y[ind[RESERVE]] @ y[indc]) * self.kernel(X[:, ind[RESERVE]], X[:, indc], type, scale)

        Qcc = self.kernel(X[:, indc], X[:, indc], type, scale) + deps

        converged = 0
        while not converged:

            perturbations = perturbations + 1

            if num_MVs > 0:  # change in alpha_c permitted

                # compute Qc, beta and gamma
                beta = -Rs @ numpy.array([[y[indc]], [Qc[MARGIN]]])
                gamma = numpy.zeros((Q.shape[1], 1))
                ind_temp = numpy.array([ind[ERROR], ind[RESERVE], indc])
                gamma[ind_temp] = [[Qc[ERROR]], [Qc[RESERVE]], [Qcc]] + Q[:, ind_temp].T @ beta

                # check if gamma_c < 0 (kernel matrix is not positive semi-definite)
                if gamma[indc] < 0:
                    print('LEARN: gamma_c < 0')

            else:  # change in alpha_c not permitted since the constraint on the sum of the
                # alphas must be preserved.  only b can change.

                # set beta and gamma
                beta = y[indc]
                gamma = y[indc] * y # @

            # minimum acceptable parameter change (change in alpha_c (num_MVs > 0) or b (num_MVs = 0))
            min_delta_param, indss, cstatus, nstatus = self.min_delta_acb(indc, gamma, beta, 1, rflag)

            # update a, b, and g
            if num_MVs > 0:
                a[indc] = a[indc] + min_delta_param
                a[ind[MARGIN]] = a[ind[MARGIN]] + beta[1:(num_MVs+1)] * min_delta_param

            b = b + beta[0] @ min_delta_param
            g = g + gamma @ min_delta_param

            # update Qc and perform bookkeeping
            converged = (indss == indc)
            if converged:
                cstatus = UNLEARNED
                Qc[nstatus] = numpy.array([[Qc[nstatus]], [Qcc]])
            else:
                ind_temp = ind[cstatus] == indss
                Qc[nstatus] = numpy.array([[Qc[nstatus]], [Qc[nstatus][ind_temp]]])
                Qc[nstatus][ind_temp] = []

            indco, removed_i = self.bookkeeping(indss, cstatus, nstatus)
            if (nstatus == RESERVE) & (removed_i > 0):
                Qc[nstatus][removed_i] = []

            # set g(ind{MARGIN}) to zero
            g[ind[MARGIN]] = 0
    
            # update Rs and Q if necessary
            if nstatus == MARGIN:
    
                num_MVs = num_MVs + 1
                if num_MVs > 1:
                    if converged:
                        gamma = gamma[indss]
                    else:
    
                        # compute beta and gamma for indss
                        beta = -Rs @ Q[:, indss]
                        gamma = self.kernel(X[:, indss], X[:, indss], type, scale) + deps + Q[:, indss].T @ beta
    
                # expand Rs and Q
                self.updateRQ(beta, gamma, indss)
    
            elif cstatus == MARGIN:
    
                # compress Rs and Q
                num_MVs = num_MVs - 1
                self.updateRQ(indc=indco)

        self.a = a
        self.b = b
        self.g = g

        return nstatus

    # KEVALS - Returns the number of kernel evaluations
    #
    # Syntax: kernel_evals = kevals();
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #

    def kevals(self):

        kernel_evals = self.kernel_evals

        if numpy.all(kernel_evals == 0):
            kernel_evals = 0

        self.kernel_evals = kernel_evals

        return kernel_evals

    # SVMTRAIN - Trains a support vector machine incrementally
    #            using the L1 soft margin approach developed by
    #            Cauwenberghs for two-class problems.
    #
    # Syntax: [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale)
    #         [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale,uind)
    #         (trains a new SVM on the given examples)
    #
    #         [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C)
    #         [a,b,g,ind,uind,X_mer,y_mar,Rs,Q] = svmtrain(X,y,C,uind)
    #         (trains the current SVM in memory on the given examples)
    #
    #      a: alpha coefficients
    #      b: bias
    #      g: partial derivatives of cost function w.r.t. alpha coefficients
    #    ind: cell array containing indices of margin, error and reserve vectors
    #         ind{1}: indices of margin vectors
    #         ind{2}: indices of error vectors
    #         ind{3}: indices of reserve vectors
    #   uind: column vector of user-defined example indices (used for unlearning specified examples)
    #  X_mer: matrix of margin, error and reserve vectors stored columnwise
    #  y_mer: column vector of class labels (-1/+1) for margin, error and reserve vectors
    #     Rs: inverse of extended kernel matrix for margin vectors
    #      Q: extended kernel matrix for all vectors
    #      X: matrix of training vectors stored columnwise
    #      y: column vector of class labels (-1/+1) for training vectors
    #      C: soft-margin regularization parameter(s)
    #         dimensionality of C       assumption
    #         1-dimensional vector      universal regularization parameter
    #         2-dimensional vector      class-conditional regularization parameters (-1/+1)
    #         n-dimensional vector      regularization parameter per example
    #         (where n = # of examples)
    #   type: kernel type
    #           1: linear kernel        K(x,y) = x'*y
    #         2-4: polynomial kernel    K(x,y) = (scale*x'*y + 1)^type
    #           5: Gaussian kernel with variance 1/(2*scale)
    #  scale: kernel scale
    #
    # Version 3.22e -- Comments to diehl@alumni.cmu.edu
    #
    def svmtrain(self, X_new, y_new, C_new, type_new=None, scale_new=None, uind_new=None):
        # returns [a,b,g,ind,uind,X,y,Rs,Q]
        # flags for example state
        MARGIN = 0
        ERROR = 1
        RESERVE = 2
        UNLEARNED = 3

        # create a vector containing the regularization parameter
        # for each example if necessary
        if len(C_new) == 1:  # same regularization parameter for all examples
            C_new = C_new * numpy.ones(y_new.shape)
        elif len(C_new) == 2:  # class-conditional regularization parameters
            flags = (y_new == -1)
            C_new = C_new[0] * flags + C_new[1] * (not flags)

        # define arguments
        if uind_new is not None:
            uind_new = numpy.zeros(y_new.shape)

        new_model = (type_new is not None) and (scale_new is not None)

        # define global variables
        def read_global_vars():
            a = self.a  # alpha coefficients
            b = self.b  # bias
            C = self.C  # regularization parameters
            deps = self.deps  # jitter factor in kernel matrix
            g = self.g  # partial derivatives of cost function w.r.t. alpha coefficients
            ind = self.ind  # cell array containing indices of margin, error, reserve and unlearned vectors
            kernel_evals = self.kernel_evals  # kernel evaluations
            max_reserve_vectors = self.max_reserve_vectors  # maximum number of reserve vectors stored
            perturbations = self.perturbations  # number of perturbations
            Q = self.Q  # extended kernel matrix for all vectors
            Rs = self.Rs  # inverse of extended kernel matrix for margin vectors
            scale = self.scale  # kernel scale
            type = self.type  # kernel type
            uind = self.uind  # user-defined example indices
            X = self.X  # matrix of margin, error, reserve and unlearned vectors stored columnwise
            y = self.y  # column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors

            return a,b,C,deps,g,ind,kernel_evals,max_reserve_vectors,perturbations,Q,Rs,scale,type,uind,X,y

        def write_global_vars():
            self.a = a  # alpha coefficients
            self.b = b  # bias
            self.C = C  # regularization parameters
            self.deps = deps  # jitter factor in kernel matrix
            self.g = g  # partial derivatives of cost function w.r.t. alpha coefficients
            self.ind = ind  # cell array containing indices of margin, error, reserve and unlearned vectors
            self.kernel_evals = kernel_evals  # kernel evaluations
            self.max_reserve_vectors = max_reserve_vectors  # maximum number of reserve vectors stored
            self.perturbations = perturbations  # number of perturbations
            self.Q = Q  # extended kernel matrix for all vectors
            self.Rs = Rs  # inverse of extended kernel matrix for margin vectors
            self.scale = scale  # kernel scale
            self.type = type  # kernel type
            self.uind = uind  # user-defined example indices
            self.X = X  # matrix of margin, error, reserve and unlearned vectors stored columnwise
            self.y = y  # column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors

        read_global_vars()

        # initialize variables
        deps = 1e-3
        max_reserve_vectors = 3000

        if new_model:

            num_examples = X_new.shape[1]

            a = numpy.zeros((num_examples, 1))
            b = 0
            C = C_new
            g = numpy.zeros((num_examples, 1))
            ind = {}
            ind[UNLEARNED] = numpy.arange(0, num_examples)
            kernel_evals = 0
            perturbations = 0
            Q = y_new.T
            Rs = numpy.inf
            scale = scale_new
            type = type_new
            uind = uind_new
            X = X_new
            y = y_new

        else:

            num_examples = X.shape[1]
            num_new_examples = X_new.shape[1]

            a = numpy.array([[a], [numpy.zeros((num_new_examples, 1))]])
            C = numpy.array([[C], [C_new]])
            g = [[g], [numpy.zeros(num_new_examples, 1)]]
            ind[UNLEARNED] = numpy.arange(0, num_new_examples) + num_examples

            # assumes currently that there are no duplicate examples in the data - may not necessarily be true!
            Q_new = numpy.array([[y_new.T],[(y[ind[MARGIN]] @ y_new.T) * self.kernel(X[:, ind[MARGIN]], X_new, type, scale)]])

            Q = numpy.array([Q, Q_new])
            uind = numpy.array([[uind], [uind_new]])
            X = numpy.array([X, X_new])
            y = numpy.array([[y], [y_new]])

            num_examples = num_examples + num_new_examples

        # begin incremental learning - enforce all constraints on each iteration
        num_learned = 1
        print('Beginning training.')

        while numpy.any(ind[UNLEARNED]):

            # randomly select example
            i = round(random.random() * (len(ind[UNLEARNED]) - 1)) + 1
            indc = ind[UNLEARNED][i]
            #  indc = ind[UNLEARNED](1)

            write_global_vars()

            # learn example
            self.learn(indc, 1)

            read_global_vars()

            if num_learned % 50 == 0:
                print('Learned %d examples.' % num_learned)

            num_learned = num_learned + 1

        if (num_learned - 1) % 50 != 0:
            print('Learned %d examples.' % (num_learned - 1))

        print('Training complete!')

        # begin incremental learning - perform multiple passes through the data
        # until all of the examples are learned
        # while (any(ind{UNLEARNED}))
        #   while (any(ind{UNLEARNED}))
        #
        #      # select example
        #      indc = ind{UNLEARNED}(1)
        #
        #      # learn example
        #      s = sprintf('\nLearning example #d...',indc)
        #      disp(s)
        #      learn(indc,0)
        #
        #   end
        #
        #   # check to see if any reserve vectors are incorrectly classified
        #   # if so, change their status to unlearned
        #   ind_temp = find(g(ind{RESERVE}) < 0)
        #   [ind{RESERVE},ind{UNLEARNED}] = move_ind(ind{RESERVE},ind{UNLEARNED},ind{RESERVE}(ind_temp))
        #
        # end

        # remove all but the closest reserve vectors from the dataset if necessary
        if len(ind[RESERVE]) == max_reserve_vectors:
            ind_keep = numpy.array([ind[MARGIN], ind[ERROR], ind[RESERVE]])
            a = a[ind_keep]
            g = g[ind_keep]
            Q = Q[:, ind_keep]
            uind = uind[ind_keep]
            X = X[:, ind_keep]
            y = y[ind_keep]
            ind[MARGIN] = numpy.arange(0, len(ind[MARGIN]))
            ind[ERROR] = len(ind[MARGIN]) + numpy.arange(0, len(ind[ERROR]))
            ind[RESERVE] = len(ind[MARGIN]) + len(ind[ERROR]) + numpy.arange(0, len(ind[RESERVE]))

        # summary statistics
        print('\nMargin vectors:\t\t%d' % len(ind[MARGIN]))
        print('Error vectors:\t\t%d' % len(ind[ERROR]))
        print('Reserve vectors:\t%d' % len(ind[RESERVE]))
        print('Kernel evaluations:\t%d\n' % self.kevals())

        write_global_vars()

        return a, b, g, ind, uind, X, y, Rs, Q

    def fit(self, X, T):

        X = X.T
        T = T.T
        _, _, _, _, _, _, output, _, _ = self.svmtrain(X, T, numpy.array([10]), 5, 2)
        output = output.T
        return output

    def predict(self, X):

        X = X.T
        output, _ = self.svmeval(X)
        output = output.T
        return output

    def fit_labeled(self, xi, yi):

        return

    def fit_unlabeled(self, xi):

        return

    def fit_unlabeled_batch(self, X):

        return

class IncrementalSVMnclasses:
    def __init__(self):
        self.nclasses = 2
        self.isvms = []

    def fit(self, X, T):
        u = numpy.unique(T)
        self.nclasses = u.shape[0]

        for c in range(0, self.nclasses):
            isc = (T == c)
            ic = numpy.nonzero(isc)
            isnotc = numpy.invert(isc)
            inc = numpy.nonzero(isnotc)
            Tc = T
            Tc[ic] = 1
            Tc[inc] = -1
            isvm = IncrementalSVM()
            isvm.fit(X, Tc)
            self.isvms.append(isvm)

        return self.nclasses

    def predict(self, X):
        m = X.shape[0]
        allres = numpy.zeros((m, self.nclasses))

        for c in range(0, self.nclasses):
            res = self.isvms[c].predict(X)
            allres[:, c] = res.T

        output = numpy.argmax(allres, axis=1)

        return output

    def fit_labeled(self, xi, yi):

        return

    def fit_unlabeled(self, xi):

        return

    def fit_unlabeled_batch(self, X):

        return
