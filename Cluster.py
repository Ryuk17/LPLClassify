#coding=utf-8
"""
@FileName: Cluster.py
@Description: Implement Cluster
@Author: Ryuk
@CreateDate: 2020/05/26
@LastEditTime: 2020/05/26
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

def euclidean(x0, x1):
    x0, x1 = np.array(x0), np.array(x1)
    d = np.sum((x0 - x1)**2)**0.5
    return d

def scaledown(X, distance=euclidean, rate=0.1, itera=1000, rand_time=10, verbose=1):
    n = len(X)

    # calculate distances martix in high dimensional space
    realdist = np.array([[distance(X[i], X[j]) for j in range(n)] for i in range(n)])
    realdist = realdist / np.max(realdist)  # rescale between 0-1

    minerror = None
    for i in range(rand_time):  # search for n times

        if verbose: print("%s/%s, min_error=%s" % (i, rand_time, minerror))

        # initilalize location in 2-D plane randomly
        loc = np.array([[np.random.random(), np.random.random()] for i in range(n)])

        # start iterating
        lasterror = None
        for m in range(itera):

            # calculate distance in 2D plane
            fakedist = np.array([[np.sum((loc[i] - loc[j]) ** 2) ** 0.5 for j in range(n)] for i in range(n)])

            # calculate move step
            movestep = np.zeros_like(loc)
            total_error = 0
            for i in range(n):
                for j in range(n):
                    if realdist[i, j] <= 0.01: continue
                    error_rate = (fakedist[i, j] - realdist[i, j]) / fakedist[i, j]
                    movestep[i, 0] += ((loc[i, 0] - loc[j, 0]) / fakedist[i, j]) * error_rate
                    movestep[i, 1] += ((loc[i, 1] - loc[j, 1]) / fakedist[i, j]) * error_rate
                    total_error += abs(error_rate)

            if lasterror and total_error > lasterror: break  # stop iterating if error becomes worse
            lasterror = total_error

            # update location
            loc -= rate * movestep

        # save best location
        if minerror is None or lasterror < minerror:
            minerror = lasterror
            best_loc = loc

    return best_loc


position = ['top', 'jug', 'mid', 'bot', 'sup']

csv_name = "./data/%s.csv" % position[0]
data = pd.read_csv(csv_name, encoding='utf-8')

data = data[data.出场次数 > 10]
ID = data.pop('ID').tolist()
data.pop('出场次数')

features = np.array(data)
scaler = StandardScaler()
norm_features = scaler.fit_transform(features)

y_pred = KMeans(n_clusters=5).fit_predict(norm_features)
print(y_pred)

# draw
loc = scaledown(norm_features, itera=1000, rand_time=10, verbose=0)

x = loc[:,0]
y = loc[:,1]
c = y_pred

fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(x, y, c=c, s=50, marker='o')
for i in range(len(x)):
    plt.annotate(ID[i], xy = (x[i], y[i]), xytext = (x[i]+0.02, y[i]+0.01))
ax.set_title('Top聚类结果')
plt.savefig('./pic/top.png')
plt.show()

