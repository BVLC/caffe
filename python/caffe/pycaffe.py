"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from ._caffe import CaffeNet, SGDSolver
from collections import OrderedDict

class Net(CaffeNet):
    """
    The direct Python interface to caffe, exposing Forward and Backward
    passes, data, gradients, and layer parameters
    """
    def __init__(self, param_file, pretrained_param_file):
        super(Net, self).__init__(param_file, pretrained_param_file)
        self._blobs = OrderedDict([(bl.name, bl)
                                   for bl in super(Net, self).blobs])
        self.params = OrderedDict([(lr.name, lr.blobs)
                                   for lr in super(Net, self).layers
                                   if len(lr.blobs) > 0])

    @property
    def blobs(self):
        """
        An OrderedDict (bottom to top, i.e., input to output) of network
        blobs indexed by name
        """
        return self._blobs
