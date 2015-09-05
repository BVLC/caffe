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

class TheanoGPU(PyLayer):
    def setup(self, bottom, top):
        import theano.tensor as T
        import theano
        from theano.sandbox.cuda.basic_ops import gpu_from_host
        x = []
        for i in range(len(bottom)):
            if len(bottom[i].shape) == 1:
                x.append(T.vector('x%d' % i))
            if len(bottom[i].shape) == 2:
                x.append(T.matrix('x%d' % i))
            if len(bottom[i].shape) == 3:
                x.append(T.tensor3('x%d' % i))
            if len(bottom[i].shape) == 4:
                x.append(T.tensor4('x%d' % i))
        y = eval(self.pythonargs['function'])
        self.f = theano.function(x, gpu_from_host(y), on_unused_input='ignore')

        if len(self.pythonargs['top_shape']) == 1:
            v = T.vector('v')
        elif len(self.pythonargs['top_shape']) == 2:
            v = T.matrix('v')
        elif len(self.pythonargs['top_shape']) == 3:
            v = T.tensor3('v')
        elif len(self.pythonargs['top_shape']) == 4:
            v = T.tensor4('v')
        self.b = []
        for i in range(len(bottom)):
            yg = T.Lop(y, x[i], v)
            self.b = theano.function(x + [v], gpu_from_host(yg), on_unused_input='ignore')
    def forward(self, bottom, top):
        from theano.misc.pycuda_utils import to_gpuarray
        top[0].reshape(self.pythonargs['top_shape'])
        t = top[0].data_tensor.to_gpuarray()
        t -= t
        tbottoms = []
        for b in bottom:
            tbottoms.append(b.data_tensor.to_cudandarray())
        output = self.f(*tbottoms)
        result = to_gpuarray(output)
        if t.shape != result.shape:
            raise ValueError('shape mismatch: %s != %s' % (t.shape, result.shape))
        t += result
    def backward(self, top, bottom):
        from theano.misc.pycuda_utils import to_gpuarray
        tdiff = top[0].diff_tensor.to_cudandarray()
        bottom_data = []
        for b in bottom:
            bottom_data.append(b.data_tensor.to_cudandarray())
        for b in bottom:
            output = self.b(*(bottom_data + [tdiff]))
            a = b.diff_tensor.to_gpuarray()
            a += to_gpuarray(output)
