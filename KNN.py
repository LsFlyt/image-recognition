#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cifar10

import numpy as np

N_DATA = 50000
N_TEST = 100
N_KIND = 10

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
	
def get_pic():
    file_name = 'data_batch_'
    tmp = unpickle(file_name+'1')
    pic_list = tmp[b'data']
    pic_labels = tmp[b'labels']
    for x in ['2','3','4','5']:
        tmp = unpickle(file_name+x)
        pic_list = np.concatenate([pic_list,tmp[b'data']],axis = 0)
        pic_labels = pic_labels + tmp[b'labels']
    return pic_list, pic_labels

def get_dist(a, b):
    ax = np.array(a[0:1024]).reshape(32,32)
    ay = np.array(a[1024:2048]).reshape(32,32)
    az = np.array(a[2048:3072]).reshape(32,32)
    bx = np.array(b[0:1024]).reshape(32,32)
    by = np.array(b[1024:2048]).reshape(32,32)
    bz = np.array(b[2048:3072]).reshape(32,32)
    return (np.sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz))).sum()
    
data_list, data_labels = get_pic()
tmp = unpickle("test_batch")
test_list = tmp[b'data']
test_labels = tmp[b'labels']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

K = 5
cnt = 0;

for x in range(N_TEST):
    L = [-1] * K;
    dist = [-1.0] * K
    for i in range(N_DATA):
        tmp = get_dist(test_list[x], data_list[i])
        for j in range(K):
            if L[j] == -1 or dist[j] > tmp:
                for l in range(K-1, j, -1):
                    L[l] = L[l-1]
                    dist[l] = dist[l-1]
                L[j] = i
                dist[j] = tmp
                break;
    now = 0
    s = [0] * N_KIND
    for i in range(K):
        s[data_labels[L[i]]] = s[data_labels[L[i]]] + 1
        if s[now] < s[data_labels[L[i]]]:
            now = data_labels[L[i]]
    print(x)
    if now == test_labels[x]:
        cnt = cnt + 1
        
print(cnt, N_TEST)