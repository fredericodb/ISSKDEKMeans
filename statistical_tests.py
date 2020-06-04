import csv

# Wilcoxon signed-rank test
from numpy.random import seed
import numpy as np
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
# read results from file
skm_res = list(csv.reader(open('skm_results.txt', 'r'), delimiter='\t'))
algs = skm_res[0]
skm_res.remove(algs)
lp = np.asarray([float(skm_res[i][0]) for i in range(16)])
mf = np.asarray([float(skm_res[i][1]) for i in range(16)])
lpp = np.asarray([float(skm_res[i][2]) for i in range(16)])
elm = np.asarray([float(skm_res[i][3]) for i in range(16)])
sgd = np.asarray([float(skm_res[i][4]) for i in range(16)])
gnb = np.asarray([float(skm_res[i][5]) for i in range(16)])
osgwr = np.asarray([float(skm_res[i][6]) for i in range(16)])
skm = np.asarray([float(skm_res[i][7]) for i in range(16)])

for i in range(0, 7):
    print('skm x %s' % algs[i])
    # compare samples
    stat, p = wilcoxon(skm, lp)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')