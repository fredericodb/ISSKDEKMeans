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
res = []
res.append(np.asarray([float(skm_res[i][0]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][1]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][2]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][3]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][4]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][5]) for i in range(16)]))
res.append(np.asarray([float(skm_res[i][6]) for i in range(16)]))
ref = np.asarray([float(skm_res[i][7]) for i in range(16)])

for i in range(0, 7):
    print('skm x %s' % algs[i])
    # compare samples
    stat, p = wilcoxon(ref, res[i])
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')