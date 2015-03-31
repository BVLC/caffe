from .pycaffe import Net, SGDSolver, LayerParameter
from ._caffe import (
    set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver,
    create_layer,
)
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
import io
