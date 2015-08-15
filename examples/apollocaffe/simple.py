import apollocaffe
from apollocaffe.layers import NumpyData, Convolution, EuclideanLoss
import numpy as np

net = apollocaffe.ApolloNet()
for i in range(1000):
    example = np.array(np.random.random()).reshape((1, 1, 1, 1))
    net.clear_forward()
    net.f(NumpyData('data', example))
    net.f(NumpyData('label', example*3))
    net.f(Convolution('conv', (1,1), 1, bottoms=['data']))
    net.f(EuclideanLoss('loss', bottoms=['conv', 'label']))
    net.backward()
    net.update(lr=0.1)
    if i % 100 == 0:
        print net.loss
