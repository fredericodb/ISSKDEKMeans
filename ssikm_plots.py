import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

colors = ['red','green','blue']  # R -> G -> B
# Create the colormap
cm = ListedColormap(colors)

graph = 2

if (graph == 0):
    c1x1 = np.random.rand(50)
    c1x1 = c1x1-0.5+5
    c1x2 = np.random.rand(50)
    c1x2 = c1x2-0.5+5
    c2x1 = np.random.rand(10)
    c2x1 = c2x1-0.5+4
    c2x2 = np.random.rand(10)
    c2x2 = c2x2-0.5+4
    c3x1 = np.random.rand(6)
    c3x1 = c3x1-0.5+4
    c3x2 = np.random.rand(6)
    c3x2 = c3x2-0.5+3
    m1x1 = np.mean(c1x1)
    m1x2 = np.mean(c1x2)
    m2x1 = np.mean(c2x1)
    m2x2 = np.mean(c2x2)
    m3x1 = np.mean(c3x1)
    m3x2 = np.mean(c3x2)
elif (graph == 1):
    c1x1 = np.random.rand(50)
    c1x1 = c1x1 - 0.5 + 5
    c1x2 = np.random.rand(50)
    c1x2 = c1x2 - 0.5 + 5
    c2x1 = np.random.rand(10)
    c2x1 = c2x1 - 0.5 + 4
    c2x2 = np.random.rand(10)
    c2x2 = c2x2 - 0.5 + 4
    c3x1 = np.random.rand(6)
    c3x1 = c3x1 - 0.5 + 4
    c3x2 = np.random.rand(6)
    c3x2 = c3x2 - 0.5 + 3.5
    m1x1 = np.mean(c1x1)
    m1x2 = np.mean(c1x2)
    m2x1 = np.mean(c2x1)
    m2x2 = np.mean(c2x2)
    m3x1 = np.mean(c3x1)
    m3x2 = np.mean(c3x2)
elif (graph == 2):
    c1x1 = np.random.rand(50)
    c1x1 = c1x1 - 0.5 + 5
    c1x2 = np.random.rand(50)
    c1x2 = c1x2 - 0.5 + 5
    c2x1 = np.random.rand(10)
    c2x1 = c2x1 - 0.5 + 4.5
    c2x2 = np.random.rand(10)
    c2x2 = c2x2 - 0.5 + 4.5
    c3x1 = np.random.rand(6)
    c3x1 = c3x1 - 0.5 + 4
    c3x2 = np.random.rand(6)
    c3x2 = c3x2 - 0.5 + 3.5
    seeds = np.vstack((np.vstack((c1x1[0],c1x2[0])).T, np.vstack((c2x1[0],c2x2[0])).T, np.vstack((c3x1[0],c3x2[0])).T))

    from sklearn.cluster import KMeans
    kmu = KMeans(init=seeds, n_clusters=3, n_init=1, max_iter=100)
    c1data = np.vstack((c1x1, c1x2)).T
    c2data = np.vstack((c2x1, c2x2)).T
    c3data = np.vstack((c3x1, c3x2)).T
    data = np.vstack((c1data,c2data,c3data))
    kmu.fit(data)
    y = kmu.predict(data)
    c1c = np.zeros((50,))
    for i in range(0,50):
        c1c[i] = y[i]
    c2c = np.zeros((10,))
    for i in range(0,10):
        c2c[i] = y[i+50]
    c3c = np.zeros((6,))
    for i in range(0, 6):
        c3c[i] = y[i+60]
    m1x1 = kmu.cluster_centers_[0][0]
    m1x2 = kmu.cluster_centers_[0][1]
    m2x1 = kmu.cluster_centers_[1][0]
    m2x2 = kmu.cluster_centers_[1][1]
    m3x1 = kmu.cluster_centers_[2][0]
    m3x2 = kmu.cluster_centers_[2][1]

mx1 = [m1x1, m2x1, m3x1]
mx2 = [m1x2, m2x2, m3x2]
plt.scatter(c1x1,c1x2,c=c1c,cmap=cm,marker='o')
plt.scatter(c2x1,c2x2,c=c2c,cmap=cm,marker='^')
plt.scatter(c3x1,c3x2,c=c3c,cmap=cm,marker='s')
plt.scatter(mx1,mx2,c='k',marker='+')

sdc1x1 = np.std(c1x1)
sdc1x2 = np.std(c1x2)
sdc2x1 = np.std(c2x1)
sdc2x2 = np.std(c2x2)
sdc3x1 = np.std(c3x1)
sdc3x2 = np.std(c3x2)
sdc1 = np.sqrt(np.power(sdc1x1,2)+np.power(sdc1x2,2))
sdc2 = np.sqrt(np.power(sdc2x1,2)+np.power(sdc2x2,2))
sdc3 = np.sqrt(np.power(sdc3x1,2)+np.power(sdc3x2,2))
cr1 = plt.Circle((m1x1,m1x2),sdc1,fill=False,color='k')
cr2 = plt.Circle((m2x1,m2x2),sdc2,fill=False,color='k')
cr3 = plt.Circle((m3x1,m3x2),sdc3,fill=False,color='k')
cr1x3 = plt.Circle((m1x1,m1x2),3*sdc1,fill=False,color='k',linestyle='--')
cr2x3 = plt.Circle((m2x1,m2x2),3*sdc2,fill=False,color='k',linestyle='--')
cr3x3 = plt.Circle((m3x1,m3x2),3*sdc3,fill=False,color='k',linestyle='--')
ax = plt.gca()
ax.add_artist(cr1)
ax.add_artist(cr2)
ax.add_artist(cr3)
#ax.add_artist(cr1x3)
#ax.add_artist(cr2x3)
#ax.add_artist(cr3x3)
ax.plot()
plt.show()
