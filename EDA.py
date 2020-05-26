#coding=utf-8

"""
@FileName: EDA.py
@Description: Implement EDA
@Author: Ryuk
@CreateDate: 2020/05/26
@LastEditTime: 2020/05/26
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

position = ['top', 'jug', 'mid', 'bot', 'sup']

csv_name = "./data/%s.csv" % position[4]
data = pd.read_csv(csv_name, encoding='utf-8')

data = data[data.出场次数 > 10]
ID = data.pop('ID')
data.pop('出场次数')


cor_fig = plt.figure(figsize=(9, 10))
correlation = data.corr()
sns.heatmap(correlation, square=True)
plt.show()


kill_fig = plt.figure()
data['场均击杀'].hist(bins=5)
plt.xlabel("场均击杀")
plt.ylabel("频数")
plt.show()

gpm_fig = plt.figure()
data['GPM'].hist(bins=5)
plt.xlabel("GPM")
plt.ylabel("频数")
plt.show()

damage_fig = plt.figure()
data['输出占比'].hist(bins=5)
plt.xlabel("输出占比")
plt.ylabel("频数")
plt.show()

view_fig = plt.figure()
data['每分钟插眼数'].hist(bins=5)
plt.xlabel("每分钟插眼数")
plt.ylabel("频数")
plt.show()

dview_fig = plt.figure()
data['每分钟排眼数'].hist(bins=5)
plt.xlabel("每分钟排眼数")
plt.ylabel("频数")
plt.show()