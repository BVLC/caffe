from .pycaffe import Net, SGDSolver
from ._caffe import (
    set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver,
    get_device,
    check_mode_cpu, check_mode_gpu,
    set_random_seed,
    Blob,
)
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
import io
try:
	from ._caffe import get_cuda_num_threads, get_blocks
except ImportError:
	pass
