"""This module implements a translator model that is able to convert a network
trained using Alex Krizhevsky's cuda-convnet code to a caffe net. It is
implemented for debugging reason, and also for easier translation with layers
trained under cuda-convnet.
"""

# first of all, import the registerer
# pylint: disable=W0401
from caffe.pyutil.translator.registerer import *
from caffe.pyutil.translator.conversions import *

# In the lines below, we will import all the translators we implemented.
import translator_cmrnorm
import translator_conv
import translator_fc
import translator_neuron
import translator_pool
import translator_softmax
