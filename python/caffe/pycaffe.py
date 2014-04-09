"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from ._caffe import Net, SGDSolver
from collections import OrderedDict

# we directly update methods from Net here (rather than using composition or
# inheritance) so that nets created by caffe (e.g., by SGDSolver) will
# automatically have the improved interface

@property
def _Net_blobs(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blobs indexed by name
    """
    return OrderedDict([(bl.name, bl) for bl in self._blobs])

Net.blobs = _Net_blobs

@property
def _Net_params(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    parameters indexed by name; each is a list of multiple blobs (e.g.,
    weights and biases)
    """
    return OrderedDict([(lr.name, lr.blobs) for lr in self.layers
                                            if len(lr.blobs) > 0])

Net.params = _Net_params
