# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 00:45:09 2022

@author: Administrator
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
from scipy import linalg
from joblib import load

# load some Ica matrix and kmeans center
W = np.load("Ica_W.npy")
K = np.load("Ica_K.npy")
cluster_centers = np.load("cluster_centers.npy")
emg_mu_for_kmeans = np.load("emg_mu_for_kmeans.npy")


# some funtion for Ica
def _cube(x, fun_args):
    return x**3, (3 * x**2).mean(axis=-1)

def _gs_decorrelation(w, W, j):

    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w


def _ica_def(X, tol, g, fun_args, max_iter, w_init):

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w**2).sum())

        for i in range(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)

            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            _gs_decorrelation(w1, W, j)

            w1 /= np.sqrt((w1**2).sum())

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)
# it should be defined before online


# load FastIca model
emg_FastICA = load("emg_FastICA.joblib")

# load kmeans model
_kmeans = load("emg_kmeans.joblib")
_kmeans.fit(emg_mu_for_kmeans[0:400, [10]])


# load the new data, delete this part when doing online
p = r'1001five9.csv'
with open(p, encoding='utf-8') as f:
    data = np.loadtxt(f,delimiter = ',')  
emg_raw = data[0:64, 50000:50400]
emg_raw = emg_raw.T
print(emg_raw.shape)


# online part are as follow
start_online = time.process_time()
# extend data
df_emg_raw = pd.DataFrame(emg_raw)
emg_extended = pd.concat([df_emg_raw] + [df_emg_raw.shift(-x) for x in range(8)], axis=1).dropna()
emg_centered = emg_extended - np.mean(emg_extended, axis=0)
emg_preprocessed = emg_centered

online_time1 = time.process_time()

# start Ica part and update W
XT = emg_FastICA._validate_data(
            emg_preprocessed, copy=None, dtype=[np.float64, np.float32], ensure_min_samples=2
        ).T

X_mean = XT.mean(axis=-1)
XT -= X_mean[:, np.newaxis]
#u, d, _ = linalg.svd(XT, full_matrices=False, check_finite=False)
#del _
#K = (u / d).T[:20]  # see (6.33) p.140
#del u, d
X1 = np.dot(K, XT)
X1 *= np.sqrt(XT.shape[1])

kwargs = {
            "tol": 1e-4,
            "g": _cube,
            "fun_args": None,
            "max_iter": 200,
            "w_init": W,
        }

W, n_iter = _ica_def(X1, **kwargs)
print(XT.shape, n_iter)

component_new = np.dot(W, K)
emg_mu_new = np.dot(XT.T, component_new.T)

emg_mu_squared_new = np.square(emg_mu_new)
spike_trains_new = np.zeros_like(emg_mu_squared_new)

online_time2 = time.process_time()

# start kmeans part and update centers
for i in range(emg_mu_squared_new.shape[1]):
    _kmeans.cluster_centers_[0, 0] = cluster_centers[0, i, 0]
    _kmeans.cluster_centers_[1, 0] = cluster_centers[1, i, 0]
    
    #idx = np.argsort(_kmeans.cluster_centers_.sum(axis=1)) # 从小到大排，并提取索引
    #flag = np.zeros_like(idx)
    #flag[idx] = np.arange(len(idx)) # 默认起点0，步长1，输出数列
  
    spike_trains_new[:, i] = _kmeans.predict(emg_mu_squared_new[:, [i]])

pre_diff = pd.DataFrame(emg_mu_squared_new).diff(-1) > 0
post_diff = pd.DataFrame(emg_mu_squared_new).diff(1) > 0
spike_trains_processsed_new = spike_trains_new * pre_diff.values * post_diff.values
spike_trains_new = spike_trains_processsed_new

online_time3 = time.process_time()

# compute SIL
list_sil = []
thre_sil = 0.8
for i in range(emg_mu_squared_new.shape[1]):
    # ignore mu that has no spike trains
    print(i)
    if np.unique(spike_trains_new[:, i]).shape[0] != 2:
        list_sil.append(0)
    else:
        _sil = silhouette_score(emg_mu_squared_new[:, [i]], spike_trains_new[:, i], random_state=None)
        list_sil.append(_sil)
valid_index_mu_new = np.where(np.array(list_sil) >= thre_sil)[0].tolist()

st_valid_new = spike_trains_new[:, valid_index_mu_new]
emg_mu_valid = emg_mu_new[:, valid_index_mu_new]

end_online = time.process_time()
print("online time", end_online-online_time3, online_time3-online_time2, online_time2-online_time1, online_time1-start_online)


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

plotspikes(st_valid_new,title='neibu df')
plotspikes(spike_trains_new,title='neibu df')