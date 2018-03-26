# --------------------------------------------------------
# Math Utils
# Copyright (c) 2017 Intel
# Licensed under The MIT License [see LICENSE for details]
# Written by YAO Matrix
# --------------------------------------------------------

import numpy as np
cimport numpy as np
cimport cython
cimport openmp

from cython.parallel cimport prange
from cython.parallel cimport parallel

import os
from multiprocessing import cpu_count
cdef int thread_num = 0
try:
    thread_num = os.environ["OMP_NUM_THREADS"]
except:
    thread_num = cpu_count() / 2



@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_subtract(np.ndarray[np.float32_t, ndim=3] src, np.ndarray[np.float32_t, ndim=3] scalar):
    global thread_num

    cdef unsigned int rows = src.shape[0]
    cdef unsigned int cols = src.shape[1]
    cdef unsigned int chs = src.shape[2]

    cdef int i, j, k

    with nogil:
        for i in prange(rows, schedule = 'dynamic', num_threads = thread_num):
            for j in xrange(cols):
                for k in xrange(chs):
                    src[i, j, k] -= scalar[0, 0, k]
