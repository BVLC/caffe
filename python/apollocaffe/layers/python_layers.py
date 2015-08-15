"""
Python Layers
"""

from layer_headers import PyLayer

class SamplePythonLayer(PyLayer):
    def forward(self, bottom, top):
        print len(bottom)
        print bottom[0].data
        print 'hello'

class Double(PyLayer):
    def setup(self, bottom, top):
        print 'setting up'
    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        top[0].data_tensor.copy_from(bottom[0].data_tensor)
        top[0].data_tensor *= 2
    def backward(self, top, bottom):
        bottom[0].diff[:] += top[0].diff * 2

class LstmSequence(PyLayer):
    def setup(self, bottom, top):
        print 'setting up'
        print self.p.name
    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        top[0].data_tensor.copy_from(bottom[0].data_tensor)
        top[0].data_tensor *= 2
    def backward(self, top, bottom):
        bottom[0].diff[:] = top[0].diff * 2

class TheanoExample(PyLayer):
    def setup(self, bottom, top):
        import theano.tensor as T
        import theano
        x = T.dvector('x')
        v = T.dvector('v')
        y = x * 2
        yg = T.Lop(y, x, v)
        self.f = theano.function([x], y)
        self.b = theano.function([x, v], yg, on_unused_input='warn')
    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        bottom[0].reshape((1,))
        top[0].data[:] = self.f(bottom[0].data)
    def backward(self, bottom, top):
        top[0].reshape((1,))
        bottom[0].reshape((1,))
        bottom[0].diff[:] += self.b(bottom[0].data, top[0].diff)

