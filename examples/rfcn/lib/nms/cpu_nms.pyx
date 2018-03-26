# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np
cimport cython
cimport openmp

from cython.parallel cimport prange
from cython.parallel cimport parallel

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b

cdef inline np.int_t thresholding(np.float32_t ovr, np.float32_t thresh) nogil:
    return 1 if ovr >= thresh else 0

import os
from multiprocessing import cpu_count
cdef int set_num = 0
try:
    set_num = os.environ["OMP_NUM_THREADS"]
except:
    set_num = cpu_count() / 2

@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr
    cdef np.float32_t threshc = thresh

    global set_num

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        thread_num = set_num if (ndets - _i - 1) > set_num else (ndets - _i - 1)
        if thread_num == 0:
            continue
        with nogil:
            for _j in prange(_i + 1, ndets, schedule = 'dynamic', num_threads = thread_num):
                j = order[_j]
                if suppressed[j] == 0:
                    xx1 = max(ix1, x1[j])
                    yy1 = max(iy1, y1[j])
                    xx2 = min(ix2, x2[j])
                    yy2 = min(iy2, y2[j])
                    w = max(0.0, xx2 - xx1 + 1)
                    h = max(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    ovr = inter / (iarea + areas[j] - inter)
                    suppressed[j] = thresholding(ovr, threshc)

    return keep
