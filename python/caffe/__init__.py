import warnings

from .pycaffe import SolverParameter, NetParameter, NetState, Net, NCCL, Timer

try:
    from ._caffe import SGDSolver_half, NesterovSolver_half, \
                    AdaGradSolver_half, RMSPropSolver_half, \
                    AdaDeltaSolver_half, AdamSolver_half
except ImportError:
    warnings.warn("Caffe datatype HALF not available.")
    
try:              
    from ._caffe import SGDSolver_float, NesterovSolver_float, \
                        AdaGradSolver_float, RMSPropSolver_float, \
                        AdaDeltaSolver_float, AdamSolver_float
    from ._caffe import SGDSolver_float as SGDSolver, \
                        NesterovSolver_float as NesterovSolver, \
                        AdaGradSolver_float as AdaGradSolver, \
                        RMSPropSolver_float as RMSPropSolver, \
                        AdaDeltaSolver_float as AdaDeltaSolver, \
                        AdamSolver_float as AdamSolver
except ImportError:
    warnings.warn("Caffe datatype FLOAT not available.")

try:           
    from ._caffe import SGDSolver_double, NesterovSolver_double, \
                        AdaGradSolver_double, RMSPropSolver_double, \
                        AdaDeltaSolver_double, AdamSolver_double
except ImportError:
    warnings.warn("Caffe datatype DOUBLE not available.")

from ._caffe import init_log, log, set_mode_cpu, set_mode_gpu, set_device, \
                    Layer, set_devices, select_device, enumerate_devices, \
                    get_solver, get_solver_from_file, layer_type_list, \
                    set_random_seed, solver_count, set_solver_count, \
                    solver_rank, set_solver_rank, set_multiprocess, has_nccl, \
                    data_type, quantizer_mode
                    
from ._caffe import __version__

from .proto.caffe_pb2 import TRAIN, TEST

from .classifier import Classifier

from .detector import Detector

from . import io

from .net_spec import layers, params, NetSpec, to_proto
from .net_gen import metalayers, fix_input_dims