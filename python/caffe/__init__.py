# Suppress GLOG output for python bindings
# unless explicitly requested in environment
import os
if 'GLOG_minloglevel' not in os.environ:
    # Hide INFO and WARNING, show ERROR and FATAL
    os.environ['GLOG_minloglevel'] = '2'
    _unset_glog_level = True
else:
    _unset_glog_level = False

from .pycaffe import Net, SGDSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io

if _unset_glog_level:
    del os.environ['GLOG_minloglevel']
