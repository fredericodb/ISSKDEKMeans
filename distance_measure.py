import numpy
import scipy.spatial.distance


def distance_measure(xc, xi, t='euclidean', Si=0, w=0):

    if   t == 'euclidean':              # euclidian
        dist = scipy.spatial.distance.euclidean(xc, xi)
    elif t == 'weighted euclidean':     # weighted euclidian
        dist = scipy.spatial.distance.wminkowski(xc, xi, p=2, w=w)
    elif t == 'average':                # average
        dist = (1 / len(xc)) * scipy.spatial.distance.euclidean(xc, xi)
    elif t == 'mahalanobis':            # mahalanobis
        dist = scipy.spatial.distance.mahalanobis(xc, xi, VI=Si)
    elif t == 'cityblock':              # manhattan
        dist = scipy.spatial.distance.cityblock(xc, xi)
    elif t == 'deltas':                 # vector deltas
        dist = abs(xc - xi)
    elif t == 'canberra':               # canberra
        dist = scipy.spatial.distance.canberra(xc, xi)
    elif t == 'braycurtis':             # braycurtis
        dist = scipy.spatial.distance.braycurtis(xc, xi)
    elif t == 'cosine':                 # cosine
        dist = scipy.spatial.distance.cosine(xc, xi)
    elif t == 'chebyshev':              # chebyshev
        dist = scipy.spatial.distance.chebyshev(xc, xi)
    elif t == 'chord':                  # chord
        dist = numpy.sqrt(2 - 2 * scipy.spatial.distance.cosine(xc, xi))
    elif t == 'czekanowski':            # czekanowski
        dist = 1 - 2 * numpy.sum(numpy.min(numpy.vstack((xc, xi)), axis=0)) / numpy.sum(xc + xi)
    elif t == 'divergence':             # divergence
        dist = numpy.sqrt((1 / len(xc)) * numpy.sum(((xc - xi) / (xc + xi)) ** 2))
    elif t == 'correlation':            # correlation
        dist = scipy.spatial.distance.correlation(xc, xi)
    elif t == 'weighted mahalanobis':   # weighted mahalanobis ...
        dist = (xc-xi)*Si*(xc-xi).T
        dist = numpy.sqrt(sum(numpy.multiply(w, dist)))
    else:
        dist = 0.0

    return dist

def distance_measures(Xc, Xi, t='euclidean', Si=0, w=0):

    if   t == 'euclidean':              # euclidian
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'weighted euclidean':     # weighted euclidian
        dist = scipy.spatial.distance.cdist(Xc, Xi, 'wminkowski', p=2., w=w)
    elif t == 'average':                # average
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: (1 / len(xc)) * scipy.spatial.distance.euclidean(xc, xi))
    elif t == 'mahalanobis':            # mahalanobis
        dist = scipy.spatial.distance.cdist(Xc, Xi, t, VI=Si)
    elif t == 'cityblock':              # manhattan
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'deltas':                 # vector deltas
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: abs(xc - xi))
    elif t == 'canberra':               # canberra
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'braycurtis':             # braycurtis
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'cosine':                 # cosine
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'chebyshev':              # chebyshev
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'chord':                  # chord
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: numpy.sqrt(2 - 2 * scipy.spatial.distance.cosine(xc, xi)))
    elif t == 'czekanowski':            # czekanowski
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: 1 - 2 * numpy.sum(numpy.min(numpy.vstack((xc, xi)), axis=0)) / numpy.sum(xc + xi))
    elif t == 'divergence':             # divergence
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: numpy.sqrt((1 / len(xc)) * numpy.sum(((xc - xi) / (xc + xi)) ** 2)))
    elif t == 'correlation':            # correlation
        dist = scipy.spatial.distance.cdist(Xc, Xi, t)
    elif t == 'weighted mahalanobis':   # weighted mahalanobis ...
        dist = scipy.spatial.distance.cdist(Xc, Xi, lambda xc, xi: numpy.sqrt(sum(numpy.multiply(w, (xc-xi)*Si*(xc-xi).T))))
    else:
        dist = numpy.zeros((Xc.shape[0], Xi.shape[0]), dtype=numpy.float64)

    return dist

def nearest_center(Xc, Xi, t='euclidean', Si=0, w=0):

    dist = distance_measures(Xc, Xi, t, Si, w)
    if not dist.size:
        print('shit')
    nc = numpy.argmin(dist, axis=0)
    nd = dist[nc]
    return nc, nd

# def nearest_centers_pareto(Xc, Xi, Si=0, w=0):

