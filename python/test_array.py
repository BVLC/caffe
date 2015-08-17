import numpy as np
import caffe
import sys

def f(a, b):
	sm = 0
	for i in range(a.shape[0]):
		sm += ((a[0]-b[0]) / (a[0]+b[0])).mean()
	return sm

A = np.random.random((16, 500, 500, 3))
B = np.random.random((16, 500, 500, 3))
a, b = caffe.Array(A.shape), caffe.Array(B.shape)
a[...] = A
b[...] = B
from time import time

t0, sm = time(), 0
for i in range(16):
	sm += f(A,B)
print( "Numpy    T", time()-t0, 'SM', sm )

t0, sm = time(), 0
for i in range(16):
	sm += f(a,b)
print( "Array(C) T", time()-t0, 'SM', sm )
del a, b

caffe.set_mode_gpu()
a, b = caffe.Array(A), caffe.Array(B)

t0, sm = time(), 0
for i in range(16):
	sm += f(a,b)
print( "Array(G) T", time()-t0, 'SM', sm )
