#!/usr/bin/env python
#-*-coding=utf-8

import sys
import numpy as np
import scipy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
import copy
import random

def prune_by_sum_min(weight, axis = 1):
    weight_sum = weight.sum(axis)
    sorted_index = np.argsort(weight_sum)
    return sorted_index

def measure_entropy2(src, dst):
    kl = scipy.stats.entropy(src, dst)
    return kl

def measure_entropy(src):
    kl = -scipy.stats.entropy(src)
    return kl

def measure_l1(src):
    l1 = sum(src)
    return l1

def zero_weight(weight, ic_num, k_size, oc_index = 0):
    index_zero = np.zeros((ic_num,k_size,k_size))
    weight[oc_index] = index_zero
    return weight

def prune_by_kl(weight, ic_num, oc_num, k_size):
    kls = []
    for index in range(0, oc_num):
        weight_copy = np.fabs(copy.deepcopy(weight[index]).flatten())
        weight_copy[weight_copy < 1e-6] = 1e-6
        kl = measure_entropy(weight_copy)
        kls.append(kl)

    sorted_index = np.argsort(kls)
    return sorted_index

def prune_by_l1(weight, ic_num, oc_num, k_size):
    kls = []
    for index in range(0, oc_num):
        weight_copy = np.fabs(copy.deepcopy(weight[index]).flatten())
        kl = measure_l1(weight_copy)
        kls.append(kl)

    sorted_index = np.argsort(kls)
    return sorted_index

def prune_by_random(weight, ic_num, oc_num, k_size):
    rds = []
    for index in range(0, oc_num):
        rds.append(random.random())

    sorted_index = np.argsort(rds)
    return sorted_index

def measure_mi(src, dst):
    mi = mutual_info_score(src, dst)
    return mi

def measure_nmi(src, dst):
    nmi = normalized_mutual_info_score(src, dst)
    return nmi

def prune_by_mi(weight, ic_num, oc_num):
    mis = []
    weight_flatten = weight.flatten()
    for index in range(0, oc_num):
        weight_copy = copy.deepcopy(weight)
        new_weight = zero_weight(weight_copy, ic_num, index)
        new_weight_flatten = new_weight.flatten()
        mi = measure_mi(weight_flatten, new_weight_flatten)
        mis.append(mi)

    sorted_index = np.argsort(mis)
    return sorted_index

def prune_by_nmi(weight, ic_num, oc_num):
    nmis = []
    weight_flatten = weight.flatten()
    for index in range(0, oc_num):
        weight_copy = copy.deepcopy(weight)
        new_weight = zero_weight(weight_copy, ic_num, index)
        new_weight_flatten = new_weight.flatten()
        nmi = measure_nmi(weight_flatten, new_weight_flatten)
        nmis.append(nmi)

    sorted_index = np.argsort(nmis)
    return sorted_index
