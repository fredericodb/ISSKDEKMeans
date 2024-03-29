import time

from readmat import readmat
import numpy
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, nf, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    xxyy = numpy.c_[xx.ravel(), yy.ravel()]
    RD = numpy.zeros((xxyy.shape[0], nf-2))
    xyr = numpy.concatenate((xxyy, RD), axis=1)
    Z = clf.predict(xyr)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

datasetsource = 'setup28'
datasetvar = 'Dataset'

dic = readmat(datasetsource, [datasetvar])

datasets = dic[datasetvar][0]
#datasets = numpy.array([['cancer'], ['horse'],['usps']])
#datasets = numpy.array([['appendicitis'], ['cleveland'],['g241n']])

accs = numpy.zeros((datasets.shape[0], 15))
ks = numpy.zeros((datasets.shape[0], 4))
times = numpy.zeros((datasets.shape[0], 15))
di = 0
for ds in datasets:
    if ds == 'lenses' or ds == 'zoo' or ds == 'digit1': # or ds == 'cleveland' or ds == 'led7digit' or ds == 'newthyroid':
       di = di+1
       continue
    print('\nBase %s\n' % ds)
    # read data samples
    dic = readmat('ds_'+ds[0]+'.mat', ['samples', 'labels'])
    samples = dic['samples']
    labels = dic['labels']

    # encode labels
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    n_c = le.classes_.shape[0]

    # shuffles data
    import numpy
    i = numpy.random.permutation(range(labels.shape[0]))
    data = samples[i,:]
    labels = labels[i].T

    # divides data in labeled, unlabeled, and test (20%, 60%, 20%)
    import sklearn.model_selection as ms
    lnc = 0
    unc = 0
    while (lnc < n_c) or (unc < n_c):
        # unlabeled
        data_m, data_u, labels_m, labels_u = ms.train_test_split(data, labels, test_size=0.60, random_state=None)
        # labeled and test
        data_l, data_t, labels_l, labels_t = ms.train_test_split(data_m, labels_m, test_size=0.50, random_state=None)
        u = numpy.unique(labels_l)
        lnc = u.shape[0]
        u = numpy.unique(labels_u)
        unc = u.shape[0]

    from sklearn.metrics import accuracy_score

    # labelpropagation

    from sklearn.semi_supervised import LabelPropagation
    label_prop_model = LabelPropagation()
    labels_fu = numpy.ones(labels_u.shape)*-1
    data_lu = numpy.concatenate((data_l, data_u), axis=0)
    labels_lu = numpy.concatenate((labels_l, labels_fu), axis=0)
    _start_time = time.time()
    label_prop_model.fit(data_lu, labels_lu)
    _end_time = time.time()
    times[di, 0] = (_end_time - _start_time)
    output_lp = label_prop_model.predict(data_t)
    error_lp = accuracy_score(labels_t, output_lp)

    # # labelspreading
    #
    # from sklearn.semi_supervised import LabelSpreading
    # label_spr_model = LabelSpreading()
    # data_lu = numpy.concatenate((data_l, data_u), axis=0)
    # labels_lu = numpy.concatenate((labels_l, labels_u), axis=0)
    # label_spr_model.fit(data_lu, labels_lu)
    # output_ls = label_spr_model.predict(data_t)
    # error_ls = accuracy_score(labels_t, output_ls)

    # # IncrementalSVMnclasses ISVM
    #
    # from incremental_svm import IncrementalSVMnclasses
    # isvm = IncrementalSVMnclasses()
    # isvm.fit(data_l, labels_l)
    # output_isvm = isvm.predict(data_t)
    # error_isvm = accuracy_score(labels_t, output_isvm)

    # Mondrian Forest MondrianForest

    def load_data(data_x, labels_x, data_tx, labels_tx):
        n_dim = data_x.shape[1]
        n_class = n_c
        x_train = data_x
        y_train = labels_x
        x_test = data_tx
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]
        y_test = labels_tx
        data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
                'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
                'y_test': y_test, 'n_test': n_test, 'is_sparse': False}

        assert (not data['is_sparse'])
        try:
            if settings.normalize_features == 1:
                min_d = numpy.minimum(numpy.min(data['x_train'], 0), numpy.min(data['x_test'], 0))
                max_d = numpy.maximum(numpy.max(data['x_train'], 0), numpy.max(data['x_test'], 0))
                range_d = max_d - min_d
                idx_range_d_small = range_d <= 0.  # find columns where all features are identical
                if data['n_dim'] > 1:
                    range_d[idx_range_d_small] = 1e-3  # non-zero value just to prevent division by 0
                elif idx_range_d_small:
                    range_d = 1e-3
                data['x_train'] -= (min_d + 0).astype(data['x_train'].dtype)
                data['x_train'] = data['x_train'].astype(dtype=numpy.float64)
                data['x_train'] /= range_d.astype(data['x_train'].dtype)
                data['x_test'] -= (min_d + 0.).astype(data['x_test'].dtype)
                data['x_test'] = data['x_test'].astype(dtype=numpy.float64)
                data['x_test'] /= range_d.astype(data['x_test'].dtype)
        except AttributeError:
            # backward compatibility with code without normalize_features argument
            pass
        # ------ beginning of hack ----------
        is_mondrianforest = True
        settings.n_minibatches = int(numpy.floor(data['n_train'] / data['n_test']) + 1)
        n_minibatches = settings.n_minibatches
        if is_mondrianforest:
            # creates data['train_ids_partition']['current'] and data['train_ids_partition']['cumulative'] 
            #    where current[idx] contains train_ids in minibatch "idx", cumulative contains train_ids in all
            #    minibatches from 0 till idx  ... can be used in gen_train_ids_mf or here (see below for idx > -1)
            data['train_ids_partition'] = {'current': {}, 'cumulative': {}}
            train_ids = numpy.arange(data['n_train'])
            try:
                draw_mondrian = settings.draw_mondrian
            except AttributeError:
                draw_mondrian = False
            if is_mondrianforest and (not draw_mondrian):
                reset_random_seed(settings)
                numpy.random.shuffle(train_ids)
                # NOTE: shuffle should be the first call after resetting random seed
                #       all experiments would NOT use the same dataset otherwise
            train_ids_cumulative = numpy.arange(0)
            n_points_per_minibatch = int(data['n_train'] / n_minibatches)
            assert n_points_per_minibatch > 0
            idx_base = numpy.arange(n_points_per_minibatch, dtype=numpy.int64)
            for idx_minibatch in range(n_minibatches):
                is_last_minibatch = (idx_minibatch == n_minibatches - 1)
                idx_tmp = idx_base + idx_minibatch * n_points_per_minibatch
                if is_last_minibatch:
                    # including the last (data[n_train'] % settings.n_minibatches) indices along with indices in idx_tmp
                    idx_tmp = numpy.arange(idx_minibatch * n_points_per_minibatch, data['n_train'], dtype=numpy.int64)
                train_ids_current = train_ids[idx_tmp]
                # print(idx_minibatch, train_ids_current
                data['train_ids_partition']['current'][idx_minibatch] = train_ids_current
                train_ids_cumulative = numpy.append(train_ids_cumulative, train_ids_current)
                data['train_ids_partition']['cumulative'][idx_minibatch] = train_ids_cumulative

        return data

    from mondrianforest import MondrianForest, process_command_line, precompute_minimal, reset_random_seed
    settings = process_command_line()
    data_lu = numpy.concatenate((data_l, data_u), axis=0)
    labels_lu = numpy.concatenate((labels_l, labels_u), axis=0)
    data_lu_mf = load_data(data_lu, labels_lu, data_l, labels_l)
    param, cache = precompute_minimal(data_lu_mf, settings)
    weights_prediction = numpy.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
    mf = MondrianForest(settings, data_lu_mf)
    train_ids_current_minibatch = data_lu_mf['train_ids_partition']['current'][0]
    _start_time = time.time()
    mf.fit(data_lu_mf, train_ids_current_minibatch, settings, param, cache)
    _end_time = time.time()
    times[di, 1] = (_end_time - _start_time)
    pred_forest_test, metrics_test = \
        mf.evaluate_predictions(data_lu_mf, data_t, labels_t, settings, param, weights_prediction, False)
    output_mf = numpy.argmax(pred_forest_test['pred_prob'], axis=1)
    error_mf = accuracy_score(labels_t, output_mf)
    times[di, 2] = 0
    for idx_batch in range(1, settings.n_minibatches):
        train_ids_current_minibatch = data_lu_mf['train_ids_partition']['current'][idx_batch]
        _start_time = time.time()
        mf.partial_fit(data_lu_mf, train_ids_current_minibatch, settings, param, cache)
        _end_time = time.time()
        times[di, 2] = times[di, 2] + (_end_time - _start_time)
    pred_forest_test, metrics_test = \
        mf.evaluate_predictions(data_lu_mf, data_t, labels_t, settings, param, weights_prediction, False)
    output_mfo = numpy.argmax(pred_forest_test['pred_prob'], axis=1)
    error_mfo = accuracy_score(labels_t, output_mfo)

    # Learn++ LearnPP

    from learnpp import LearnPP
    lpp = LearnPP()
    _start_time = time.time()
    lpp.fit(data_l, labels_l)
    _end_time = time.time()
    times[di, 3] = (_end_time - _start_time)
    output_lpp = lpp.predict(data_t)
    error_lpp = accuracy_score(labels_t, output_lpp)
    _start_time = time.time()
    lpp.partial_fit(data_u, labels_u)
    _end_time = time.time()
    times[di, 4] = (_end_time - _start_time)
    output_lppo = lpp.predict(data_t)
    error_lppo = accuracy_score(labels_t, output_lppo)

    # Incremental ELM - IELM

    from ielm import IELM
    hls = numpy.int(numpy.round((data_l.shape[1] + n_c) / 2))
    elm = IELM(nHiddenNeurons=hls)
    _start_time = time.time()
    elm.fit(data_l, labels_l)
    _end_time = time.time()
    times[di, 5] = (_end_time - _start_time)
    output_elm = elm.predict(data_t)
    error_elm = accuracy_score(labels_t, output_elm)
    _start_time = time.time()
    elm.partial_fit(data_u, labels_u)
    _end_time = time.time()
    times[di, 6] = (_end_time - _start_time)
    output_elmo = elm.predict(data_t)
    error_elmo = accuracy_score(labels_t, output_elmo)

    # Stochastic Gradient Descent (SGD)

    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier()
    _start_time = time.time()
    sgd.fit(data_l, labels_l)
    _end_time = time.time()
    times[di, 7] = (_end_time - _start_time)
    output_sgd = sgd.predict(data_t)
    error_sgd = accuracy_score(labels_t, output_sgd)
    _start_time = time.time()
    sgd.partial_fit(data_u, labels_u)
    _end_time = time.time()
    times[di, 8] = (_end_time - _start_time)
    output_sgdo = sgd.predict(data_t)
    error_sgdo = accuracy_score(labels_t, output_sgdo)

    # Gaussian Naive Bayes (GNB)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    _start_time = time.time()
    gnb.fit(data_l, labels_l)
    _end_time = time.time()
    times[di, 9] = (_end_time - _start_time)
    output_gnb = gnb.predict(data_t)
    error_gnb = accuracy_score(labels_t, output_gnb)
    _start_time = time.time()
    gnb.partial_fit(data_u, labels_u)
    _end_time = time.time()
    times[di, 10] = (_end_time - _start_time)
    output_gnbo = gnb.predict(data_t)
    error_gnbo = accuracy_score(labels_t, output_gnbo)

    # # knn
    #
    # from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(data_l, labels_l)
    # output_knn = neigh.predict(data_t)
    # error_knn = accuracy_score(labels_t, output_knn)

    # # decision tree
    #
    # from sklearn.tree import DecisionTreeClassifier
    # dt = DecisionTreeClassifier(random_state=1)
    # dt.fit(data_l, labels_l)
    # output_dt = dt.predict(data_t)
    # error_dt = accuracy_score(labels_t, output_dt)
    #
    # # mlp
    #
    # from sklearn.neural_network import MLPClassifier
    # # mean of inputs and outputs: Nh = (Ni + No)/2
    # # upper bound for no overfitting: Nh = Ns / (alpha * (Ni + No)), alpha = 2-10
    # hls = numpy.int(numpy.round((data_l.shape[1] + n_c)/2))
    # mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hls,), random_state=1)
    # mlp.fit(data_l, labels_l)
    # output_mlp = mlp.predict(data_t)
    # error_mlp = accuracy_score(labels_t, output_mlp)
    #
    # # k-means unsupervised
    # from sklearn.cluster import KMeans
    # from sklearn.metrics import adjusted_mutual_info_score
    #
    # kmu = KMeans(init='k-means++', n_clusters=n_c, n_init=1)
    # kmu.fit(data_l)
    # output_kmu = kmu.predict(data_t)
    # error_kmu = adjusted_mutual_info_score(labels_t, output_kmu)

    # seeded k-means

    # from seeded_kmeans import Seeded_KMeans
    # seedkm = Seeded_KMeans(nk=2*n_c, n=data_l.shape[1])
    # seedkm.fit(data_l, labels_l, 100, 0)
    # output_seedkm = seedkm.predict(data_t)
    # error_seedkm = accuracy_score(labels_t, output_seedkm)

    # sskm1 - k-means supervised / semi-supervised euclidean distance

    # offline training - labeled

    def fitness_sskm1(w, *args):
        data_l, labels_l, kms, td, nit = args
        nk = kms.nk
        n = kms.n

        W = w.reshape(nk,n)

        output = kms.fit(data_l, labels_l, W, td, True, nit)
        # kms.refine(X, T, nc, nit, dt, S, w, CF, ncluster);
        return 1 - accuracy_score(labels_l, output)


    def fitness_sskm1_tst(w, *args):
        data_l, labels_l, kms = args
        nk = kms.nk
        n = kms.n

        W = w.reshape(nk, n)

        kms.cf_weights = W
        output = kms.predict(data_l)

        # minimize error = 1 - accuracy
        return 1 - accuracy_score(labels_l, output)


    def find_best_kms(data_l, labels_l, td, var, add, vartp=0):
        bk = n_c
        bW = 0
        bacc = -1
        bkms = 0
        for k in range(n_c, 4 * n_c, n_c):
            from kmeans_supervised_weighted import kmeans_supervised

            print('trying nk = %s: ' % k)
            nit = 100
            thd = 0
            alpha = 0.75
            if (td == 0) | (td == 2):
                w = numpy.ones((k, data_l.shape[1]))
                kmsl = []
                if var:
                    if vartp == 0: # vary nothing
                        kms = kmeans_supervised(k, data_l.shape[1])
                        _start_time = time.time()
                        if add:
                            output = kms.fit_var_add(data_l, labels_l, w, td, True, nit, thd, alpha)
                        else:
                            output = kms.fit_var(data_l, labels_l, w, td, True, nit, thd, alpha)
                        _end_time = time.time()
                        accopt = accuracy_score(labels_l, output)
                        timeopt = (_end_time - _start_time)
                    elif vartp == 1: # vary alpha
                        accopts = numpy.zeros((11,))
                        timeopts = numpy.zeros((11,))
                        o = 0
                        alphavar = numpy.arange(0.0, 1.1, 0.1)
                        for alpha in alphavar:
                            kms = kmeans_supervised(k, data_l.shape[1])
                            _start_time = time.time()
                            if add:
                                output = kms.fit_var_add(data_l, labels_l, w, td, True, nit, thd, alpha)
                            else:
                                output = kms.fit_var(data_l, labels_l, w, td, True, nit, thd, alpha)
                            _end_time = time.time()
                            accopt = accuracy_score(labels_l, output)
                            accopts[o] = accopt
                            timeopts[o] = (_end_time - _start_time)
                            kmsl.append(kms)
                            o = o+1
                        accopt = accopts.max()
                        alpha = alphavar[accopts.argmax()]
                        kms = kmsl[accopts.argmax()]
                        timeopt = timeopts[accopts.argmax()]
                        print('acc (alpha in [0, 1]):', accopts)
                        print('offline training times (alpha in [0, 1]):', timeopts)
                    elif vartp == 2: # vary m_o
                        accopts = numpy.zeros((10,))
                        timeopts = numpy.zeros((10,))
                        o = 0
                        movar = numpy.arange(1, 21, 2)
                        for mo in movar:
                            kms = kmeans_supervised(k, data_l.shape[1])
                            _start_time = time.time()
                            if add:
                                output = kms.fit_var_add(data_l, labels_l, w, td, True, nit, thd, alpha, mo)
                            else:
                                output = kms.fit_var(data_l, labels_l, w, td, True, nit, thd, alpha, mo)
                            _end_time = time.time()
                            accopt = accuracy_score(labels_l, output)
                            accopts[o] = accopt
                            timeopts[o] = (_end_time - _start_time)
                            kmsl.append(kms)
                            o = o+1
                        accopt = accopts.max()
                        mo = movar[accopts.argmax()]
                        kms = kmsl[accopts.argmax()]
                        timeopt = timeopts[accopts.argmax()]
                        print('acc (mo in [1, 12]):', accopts)
                        print('offline training times (mo in [1, 12]):', timeopts)
                else:
                    kms = kmeans_supervised(k, data_l.shape[1])
                    _start_time = time.time()
                    output = kms.fit(data_l, labels_l, w, td, True, nit, thd)
                    _end_time = time.time()
                    accopt = accuracy_score(labels_l, output)
                w = kms.cf_weights
                # kms.refine(X, T, nc, nit, dt, S, w, CF, ncluster);
            elif (td == 1) | (td == 5):
                w = numpy.ones((k, kms.n))
                if var:
                    if add:
                        output = kms.fit_var_add(data_l, labels_l, w, td, True, nit, thd, 0.7)
                    else:
                        output = kms.fit_var(data_l, labels_l, w, td, True, nit, thd, 0.7)
                else:
                    output = kms.fit(data_l, labels_l, w, td, True, nit, thd)
                from pyswarm import pso

                lb = numpy.zeros((kms.nk, kms.n))
                lb = lb.flatten()
                ub = numpy.ones((kms.nk, kms.n)) * 25
                ub = ub.flatten()
                # w, accopt = pso(fitness_sskm1, lb, ub, args=(data_l, labels_l, kms, td, nit))
                w, accopt = pso(fitness_sskm1_tst, lb, ub, args=(data_l, labels_l, kms))
                accopt = 1 - accopt

            W = w.reshape(kms.nk, kms.n)

            print('offline training acc = %.4f\n' % accopt)
            print('offline training time: ', timeopt)
            if (bacc == -1) | (accopt > bacc):
                bacc = accopt
                times[di, 11] = timeopt
                bk = k
                bW = W
                bkms = kms
        return bkms

    bk = n_c
    bW = 0
    bacc = -1
    td = 0
    var = True
    add = True
    vartp = 0
    bkms = find_best_kms(data_l, labels_l, td, var, add, vartp)
    bk = bkms.nk
    ks[di, 0] = bkms.nk
    print('kms nk = %d' % bk)
    # bkms.fit(data_l, labels_l, bW, td, True, nit)
    # bkms.refinefit(data_l,labels_l)
    # bkms.refinefit()

    # obtain skm with 20% + 60% labeled for comparison
    from kmeans_supervised_weighted import kmeans_supervised
    nit = 100
    thd = 0
    alpha = 0.75
    w = numpy.ones((bk, data_l.shape[1]))
    gkms = kmeans_supervised(bk, data_l.shape[1])
    gkms.fit_var_add(data_lu, labels_lu, w, td, True, nit, thd, alpha)
    output_kms = gkms.predict(data_t)
    error_kms = accuracy_score(labels_t, output_kms)
    print('kms test acc = %.4f' % error_kms)

    # saving for next test of unlabeled samples incremental learning
    import copy
    kms2 = copy.deepcopy(bkms)

    # online training - unlabeled + L% of labeled
    wd = 0
    m_u = data_u.shape[0]
    varlbl = False
    if varlbl:
        bacc == -1
        accopts = numpy.zeros((11,))
        timeopts = numpy.zeros((11,))
        o = 0
        for L in numpy.arange(0.0, 1.1, 0.1):
            s_l = int(m_u * L)
            tkms = bkms
            _start_time = time.time()
            for i in range(0, m_u):
                if (s_l != 0) and (i % s_l == 0):
                    tkms.fit_labeled(data_u[i, :], labels_u[i], wd)
                else:
                    tkms.fit_unlabeled(data_u[i, :], wd)
            _end_time = time.time()
            output = tkms.predict(data_t)
            accopt = accuracy_score(labels_t, output)
            timeopt = (_end_time - _start_time)
            accopts[o] = accopt
            timeopts[o] = (_end_time - _start_time)
            o = o + 1
            if (bacc == -1) | (accopt > bacc):
                bacc = accopt
                btime = timeopt
                bkmss = tkms
        print('acc (%labeled in [0%, 100%]):', accopts)
        print('online training times (%labeled in [0%, 100%]:', timeopts)
        timeopt = btime
        error_kmss = bacc
        bkms = bkmss
    else:
        L = 100
        s_l = int(m_u * L)
        _start_time = time.time()
        for i in range(0, m_u):
            if (s_l != 0) and (i % s_l == 0):
                bkms.fit_labeled(data_u[i, :], labels_u[i], wd)
            else:
                bkms.fit_unlabeled(data_u[i, :], wd)
        _end_time = time.time()
        timeopt = (_end_time - _start_time)
        output_kmss = bkms.predict(data_t)
        error_kmss = accuracy_score(labels_t, output_kmss)

    ks[di, 1] = bkms.nk
    times[di, 12] = timeopt
    print('kmss test acc = %.4f' % error_kmss)
    print('kmss total online time = %.3f s', timeopt)

    # osgwr online supervised growing neural gas training

    from neuralgas.oss_gwr import oss_gwr
    import networkx

    osg = oss_gwr(act_thr=0.45)
    _start_time = time.time()
    osg.train(data_l, labels_l, n_epochs=20)
    _end_time = time.time()
    osg_time = (_end_time - _start_time)
    times[di, 13] = osg_time
    ks[di, 2] = networkx.classes.function.number_of_nodes(osg.G)
    output_kms2 = osg.predict(data_t)
    error_kms2 = accuracy_score(labels_t, output_kms2)
    print('osgwr acc = %s' % error_kms2)

    # ossgwr online semi-supervised growing neural gas training

    data_lu = numpy.concatenate((data_l, data_u), axis=0)
    #labels_u[:] = -1
    labels_lu = numpy.concatenate((labels_l, labels_u), axis=0)
    ossg = oss_gwr(act_thr=0.45)
    _start_time = time.time()
    ossg.train(data_lu, labels_lu, n_epochs=20)
    _end_time = time.time()
    ossg_time = (_end_time - _start_time)
    times[di, 14] = ossg_time
    ks[di, 3] = networkx.classes.function.number_of_nodes(ossg.G)
    output_kmss2 = ossg.predict(data_t)
    error_kmss2 = accuracy_score(labels_t, output_kmss2)
    print('ossgwr acc = %s' % error_kmss2)


    print('\nResults base %s' % ds)
    # print('Acc knn = %s' % error_knn)
    # print('Acc dt = %s' % error_dt)
    # print('Acc mlp = %s' % error_mlp)
    # print('Acc kmu = %s' % error_kmu)
    print('Acc lp = %s' % error_lp)
    print('Acc mf = %s' % error_mf)
    print('Acc mfo = %s' % error_mfo)
    print('Acc lpp = %s' % error_lpp)
    print('Acc lppo = %s' % error_lppo)
    print('Acc elm = %s' % error_elm)
    print('Acc elmo = %s' % error_elmo)
    print('Acc sgd = %s' % error_sgd)
    print('Acc sgdo = %s' % error_sgdo)
    print('Acc gnb = %s' % error_gnb)
    print('Acc gnbo = %s' % error_gnbo)
    print('Acc kms = %s' % error_kms)
    print('Acc kmss = %s' % error_kmss)
    print('Acc osgwr = %s' % error_kms2)
    print('Acc ossgwr = %s' % error_kmss2)

    # models = (neigh, dt, mlp, kms2, bkms, osg, ossg)
    #
    # # title for the plots
    # titles = ('k-Nearest Neighbors',
    #           'Decision Tree',
    #           'Multi-Layer Perceptron',
    #           'Supervised K-Means',
    #           'Incremental Semi-supervised K-Means',
    #           'Online Supervised Growing When Required',
    #           'Online Semi-supervised Growing When Required')

    # Set-up 2x2 grid for plotting.
    # fig, sub = plt.subplots(1, 7)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #
    # X0, X1 = data_l[:, 0], data_l[:, 1]
    # y = labels_l
    # xx, yy = make_meshgrid(X0, X1)
    # nf = data_l.shape[1]
    #
    # for clf, title, ax in zip(models, titles, sub.flatten()):
    #     plot_contours(ax, clf, xx, yy, nf, cmap=plt.cm.coolwarm, alpha=0.8)
    #     # Plot the training points
    #     ax.scatter(data_l[:, 0], data_l[:, 1], c=labels_l, cmap=plt.cm.coolwarm, edgecolors='k')
    #     # and testing points
    #     ax.scatter(data_t[:, 0], data_t[:, 1], c=labels_t, cmap=plt.cm.coolwarm, alpha=0.6, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel('X1')
    #     ax.set_ylabel('X2')
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)
    #
    # plt.show()

    accs[di, 0] = error_lp
    accs[di, 1] = error_mf
    accs[di, 2] = error_mfo
    accs[di, 3] = error_lpp
    accs[di, 4] = error_lppo
    accs[di, 5] = error_elm
    accs[di, 6] = error_elmo
    accs[di, 7] = error_sgd
    accs[di, 8] = error_sgdo
    accs[di, 9] = error_gnb
    accs[di, 10] = error_gnbo
    accs[di, 11] = error_kms
    accs[di, 12] = error_kmss
    accs[di, 13] = error_kms2
    accs[di, 14] = error_kmss2
    di = di+1

print('Dataset\tAcc lp\tAcc mf\tAcc mfo\tAcc lpp\tAcc lppo\tAcc elm\tAcc elmo\tAcc sgd\tAcc sgdo\tAcc gnb\tAcc gnbo\tAcc_kms\tAcc kmss\tAcc osgwr\tAcc ossgwr\tk_lpp\tk_lppo\tk_kms\tk_kmss\tk_osgwr\tk_ossgwr\tt lp\tt lpp\tt lppo\tt sgd\tt sgdo\tt gnb\tt gnbo\tt_kms\tt_kmss\tt_osgwr\tt_ossgwr\n')
di = 0
for ds in datasets:
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f' % (ds, accs[di, 0], accs[di, 1], accs[di, 2], accs[di, 3], accs[di, 4], accs[di, 5], accs[di, 6], accs[di, 7], accs[di, 8], accs[di, 9], accs[di, 10], accs[di, 11], accs[di, 12], accs[di, 13], accs[di, 14], ks[di, 0], ks[di, 1], ks[di, 2], ks[di, 3], times[di, 0], times[di, 1], times[di, 2], times[di, 3], times[di, 4], times[di, 5], times[di, 6], times[di, 7], times[di, 8], times[di, 9], times[di, 10], times[di, 11], times[di, 12], times[di, 13], times[di, 14]))
    di = di+1