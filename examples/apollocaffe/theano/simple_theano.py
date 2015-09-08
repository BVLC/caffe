import apollocaffe
from apollocaffe.layers import NumpyData, Wordvec, TheanoGPU, EuclideanLoss
import numpy as np

apollocaffe.set_device(0)

net = apollocaffe.ApolloNet()

for i in range(1000):
    val1 = [[-2,4,1]]
    net.clear_forward()
    net.f(NumpyData('val1', val1))
    net.f(NumpyData('wordval', [0]))
    net.f(Wordvec('vec', 3, 1, bottoms=['wordval']))
    net.f(NumpyData('cosine_target', [1]))
    net.f(NumpyData('norm_target', [2]))
    expr = 'T.dot(x[0], x[1].T) / (T.dot(x[0], x[0].T) * T.dot(x[1], x[1].T))**0.5'
    net.f(TheanoGPU('cosine', [expr, (1,1)], bottoms=['val1', 'vec']))
    expr2 = 'T.dot(x[0], x[0].T)'
    net.f(TheanoGPU('norm', [expr2, (1,1)], bottoms=['vec']))
    net.f(EuclideanLoss('loss1', bottoms=['cosine', 'cosine_target']))
    net.f(EuclideanLoss('loss2', bottoms=['norm', 'norm_target']))
    net.backward()
    net.update(lr=0.01)
    if i % 100 == 0:
        print net.loss
        print net.blobs['vec'].data
        print net.blobs['norm'].data
