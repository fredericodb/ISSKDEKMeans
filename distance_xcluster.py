def distance_xcluster (xc,xi,t,Si,w):
    import numpy
    import scipy.spatial.distance
    if   t == 0:    # euclidian
        dd = scipy.spatial.distance.euclidean(xc, xi)
    elif t == 1:    # weighted euclidian
        dd = (xc - xi)*(xc - xi).T
        dd = numpy.sqrt(sum(numpy.multiply(w, dd)))
    elif t == 2:   # mahalanobis
        dd = scipy.spatial.distance.mahalanobis(xc, xi, Si)
    elif t == 5:    # weighted mahalanobis ...
        dd = (xc-xi)*Si*(xc-xi).T
        dd = numpy.sqrt(sum(numpy.multiply(w, dd)))

    return dd
