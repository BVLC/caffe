from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver
# On Windows, evaluating python layers from the caffe executable tool
# requires using an embedded _caffe module
# so here we try to load the embedded (no dot) module first
try:
    from embedded_caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list
    from embedded_caffe import __version__
except ImportError:
    from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list
    from ._caffe import __version__

from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
