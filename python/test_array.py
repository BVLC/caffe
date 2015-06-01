import caffe
from time import time
import numpy as np

caffe.set_mode_gpu()

t0 = time()
a = caffe.Array((1000,1000))
b = caffe.Array((1000,1000))
c = caffe.Array((1000,1000))
print( "Create", time()-t0 )

# Initialize the GPU values
t0 = time()
a -= a
b *= a
print( "Alloc", time()-t0 )

# Set the initial values
t0 = time()
a[...] = 1
b[...] = 0.1
c[...] = 0
print( "Init", time()-t0 )

t0 = time()
for i in range(300):
	c[...] = 2.*a + 3.*b;
	a[...] = 0.5 * c;
np.asarray(c)
#print( np.asarray(c) )
print( "Exp", time()-t0 )
print( "  c.shape", c.shape )
print( "  np.mean(c)", np.mean(c) )
print()

a = caffe.Array((64,4092))
a[...] = 1
b = caffe.Array((4092,4092))
b[...] = 1
c = caffe.Array()
caffe.gemm( False, False, 1.0, a, b, 0.0, c )
t0 = time()
for i in range(300):
	caffe.gemm( False, False, 1.0, a, b, 1.0, c )
np.asarray(c)
#print( np.asarray(c) )
print( "gemm", time()-t0 )
print( "  c.shape", c.shape )
print( "  np.mean(c)", np.mean(c) )
print()

a = caffe.Array((8, 256, 15, 15))
a[...] = 1
b = caffe.Array((256, 256, 5, 5))
b[...] = 1
c = caffe.Array()
caffe.conv( a, b, 2, 2, 1, 1, c )
t0 = time()
for i in range(300):
	caffe.conv( a, b, 2, 2, 1, 1, c )
np.asarray(c)
#print( np.asarray(c) )
print( "conv", time()-t0 )
print( "  c.shape", c.shape )
print( "  np.mean(c)", np.mean(c) )
print()
# print( b[0,0,0] )
