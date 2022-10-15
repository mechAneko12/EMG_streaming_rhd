# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:45:40 2022

@author: ying
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import pickle
import json
import matplotlib.pyplot as plt
import time


# load the training data
p = r'1001five9.csv'
with open(p, encoding='utf-8') as f:
    data = np.loadtxt(f,delimiter = ',')  
emg_raw = data[0:64, :]
emg_raw = emg_raw.T
print(emg_raw.shape)


# extend data
df_emg_raw = pd.DataFrame(emg_raw)
emg_extended = pd.concat([df_emg_raw] + [df_emg_raw.shift(-x) for x in range(8)], axis=1).dropna()
emg_centered = emg_extended - np.mean(emg_extended, axis=0)
emg_preprocessed = emg_centered


# FastIca
cashe_name='all'
emg_FastICA = FastICA(n_components=20,
                      random_state=0,
                      max_iter=200,
                      tol=1e-4,
                      # whiten=ica_whiten,
                      fun='cube',
                      algorithm='deflation'
                      )
print('start ICA')

emg_FastICA.fit(emg_preprocessed)
emg_mu = emg_FastICA.transform(emg_preprocessed)

emg_mu_squared = np.square(emg_mu)
spike_trains = np.zeros_like(emg_mu_squared) 


# save Ica matrix
W = emg_FastICA._unmixing
K = emg_FastICA.whitening_
np.save("Ica_W.npy", W)
np.save("Ica_K.npy", K)
np.save("emg_mu_for_kmeans.npy", emg_mu_squared)


# start kmeans
print('start kmeans')
cluster_centers = np.zeros((2, 20, 1))
for i in range(emg_mu_squared.shape[1]):
    _kmeans = KMeans(n_clusters=2, max_iter=10000, random_state=None)
    _kmeans.fit(emg_mu_squared[:, [i]])
    
    idx = np.argsort(_kmeans.cluster_centers_.sum(axis=1)) # 从小到大排，并提取索引
    flag = np.zeros_like(idx)
    flag[idx] = np.arange(len(idx)) # 默认起点0，步长1，输出数列
    
    spike_trains[:, i] = flag[_kmeans.labels_]
    cluster_centers[:, i] = _kmeans.cluster_centers_

pre_diff = pd.DataFrame(emg_mu_squared).diff(-1) > 0
post_diff = pd.DataFrame(emg_mu_squared).diff(1) > 0
spike_trains_processsed = spike_trains * pre_diff.values * post_diff.values
spike_trains = spike_trains_processsed
print('end kmeans')


# save kmeans center
np.save("cluster_centers.npy", cluster_centers)


# compute SIL
list_sil = []
thre_sil = 0.8
for i in range(emg_mu_squared.shape[1]):
    # ignore mu that has no spike trains
    print(i)
    if np.unique(spike_trains[:, i]).shape[0] != 2:
        list_sil.append(0)
    else:
        _sil = silhouette_score(emg_mu_squared[:, [i]], spike_trains[:, i], random_state=None)
        list_sil.append(_sil)
valid_index_mu_ = np.where(np.array(list_sil) >= thre_sil)[0].tolist()


# remove noise
st_valid = spike_trains[:, valid_index_mu_]
emg_mu_valid = emg_mu[:, valid_index_mu_]


# plot
def plotspikes(spikes,
                win=None,
                title='title'):
    if win is not None:
        spikes = spikes[win:]
    x_s = (np.arange(spikes.shape[0]) ) /2000
    cmap = ['#de3c3a', '#de9c3a', '#d8de3a', '#70de3a', '#3ade79', '#3adedb', '#3a7bde', '#633ade', '#d33ade', '#de3a86']# plt.get_cmap("tab10")
    i_c = 0
    for i in range(spikes.shape[1]):
        spike_row = spikes[:, i]
        if i_c >= len(cmap):
            i_c = 0
        color = cmap[i_c]
        i_c += 1
        for j in range(spikes.shape[0]):
            if spike_row[j] == 1:
                _x = x_s[j]
                plt.plot([_x, _x], [i+0.1, i+0.9], color=color, lw=1)
    plt.title(title)
    plt.xlabel('time[s]')
    # plt.ylabel('')
    plt.show()

plotspikes(st_valid,title='neibu df')
plotspikes(spike_trains,title='neibu df')


from joblib import dump
dump(emg_FastICA, "emg_FastICA.joblib")
dump(_kmeans, "emg_kmeans.joblib")