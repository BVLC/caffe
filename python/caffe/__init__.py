from .pycaffe import Net, SGDSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, \
    set_phase_train, set_phase_test
from .classifier import Classifier
from .detector import Detector
import io
