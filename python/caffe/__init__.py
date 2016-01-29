from .pycaffe import SolverParameter, Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, set_devices, select_device, enumerate_devices, Layer, get_solver, get_solver_from_file, layer_type_list
from ._caffe import __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
