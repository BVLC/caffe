from ._caffe import Blob
# Enable PyCuda only if Cafffe is bulilt with GPU
pycuda_available = False
if hasattr(Blob, 'gpu_data_ptr'):
	try:
		import pycuda.gpuarray
		pycuda_available = True
		from _pycuda_util import *
	except ImportError:
		pass
