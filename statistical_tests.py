import csv

# Wilcoxon signed-rank test
from numpy.random import seed
import numpy as np
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
################# SUPERVISED VERSION SKM #################
# read results from file
skm_res = list(csv.reader(open('skm_results.txt', 'r'), delimiter='\t'))
algs = skm_res[0]
skm_res.remove(algs)
res = []
for j in range(7):
    res.append(np.asarray([float(skm_res[i][j]) for i in range(16)]))
ref = np.asarray([float(skm_res[i][7]) for i in range(16)])
print('Wilcoxon for SKM x all')
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

################# SEMI-SUPERVISED INCREMENTAL VERSION SSIKM #################
# read results from file
skm_res = list(csv.reader(open('ssikm_results.txt', 'r'), delimiter='\t'))
algs = skm_res[0]
skm_res.remove(algs)
res = []
for j in range(7):
    res.append(np.asarray([float(skm_res[i][j]) for i in range(16)]))
ref = np.asarray([float(skm_res[i][7]) for i in range(16)])
print('Wilcoxon for SSIKM x all')
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

################# ALL VERSIONS SKM SSIKM #################
# Friedman test
from numpy.random import seed
from scipy.stats import friedmanchisquare
# seed the random number generator
seed(1)
# read results from file
skm_res = list(csv.reader(open('all_results.txt', 'r'), delimiter='\t'))
algs = skm_res[0]
skm_res.remove(algs)
res = []
for j in range(13):
    res.append(np.asarray([float(skm_res[i][j]) for i in range(16)]))
print('Friedman for all x all')
# compare samples
stat, p = friedmanchisquare(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10],
                            res[11], res[12])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')