import csv

# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
# read results from file
skm_res = list(csv.reader(open('skm_results.txt', 'r'), delimiter='\t'))
algs = skm_res[0]
lp = skm_res[1:, 0]
mf = skm_res[1:, 1]
lpp = skm_res[1:, 2]
elm = skm_res[1:, 3]
sgd = skm_res[1:, 4]
gnb = skm_res[1:, 5]
osgwr = skm_res[1:, 6]
skm = skm_res[1:, 7]

for i in range(0, 7):
    print('skm x %s', algs[i])
    # compare samples
    stat, p = wilcoxon(skm, lp)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')