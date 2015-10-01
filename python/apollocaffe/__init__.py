from cpp._apollocaffe import Tensor, ApolloNet, CppConfig, make_numpy_data_param, Blob
import loggers
import utils
from time import strftime

def set_cpp_loglevel(loglevel):
    CppConfig.set_cpp_loglevel(loglevel)

def set_device(device_id=-1):
    if device_id == -1:
        device_string = "CPU device"
    else:
        device_string = "GPU device %d" % device_id
    print("%s - %s" % \
        (strftime("%Y-%m-%d %H:%M:%S"), device_string))
    CppConfig.set_device(device_id)

def set_random_seed(value):
    import numpy as np
    import random
    np.random.seed(value)
    random.seed(value)
    CppConfig.set_random_seed(value)

def base_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--loglevel', default=3, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    return parser

set_cpp_loglevel(3)

# Apollocaffe uses print() over the obscure python logging module
# Disable buffering of stdout ,equivalent to export PYTHONUNBUFFERED=x
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)
