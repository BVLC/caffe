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
        self.top_shape = (1,)
        self.function_str = ''
    def parse_args(self, bottom, top):
        function_str = self.pythonargs[0]
        top_shape = self.pythonargs[1]

        if self.function_str != function_str or self.top_shape != top_shape:
            self.function_str = function_str
            self.top_shape = top_shape

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

            y = eval(function_str)
            self.f = theano.function(x, gpu_from_host(y), on_unused_input='ignore')

            if len(self.top_shape) == 1:
                v = T.vector('v')
            elif len(self.top_shape) == 2:
                v = T.matrix('v')
            elif len(self.top_shape) == 3:
                v = T.tensor3('v')
            elif len(self.top_shape) == 4:
                v = T.tensor4('v')
            self.b = []
            for i in range(len(bottom)):
                yg = T.Lop(y, x[i], v)
                self.b.append(theano.function(x + [v], gpu_from_host(yg), on_unused_input='ignore'))

    def forward(self, bottom, top):
        self.parse_args(bottom, top)
        top[0].reshape(self.top_shape)
        tbottoms = []
        for b in bottom:
            tbottoms.append(b.data_tensor.to_cudandarray())
        output = self.f(*tbottoms)
        top[0].data_tensor.set_values(0.)
        top[0].data_tensor.add_from_cudandarray(output)
    def backward(self, top, bottom):
        tdiff = top[0].diff_tensor.to_cudandarray()
        bottom_data = []
        for i in range(len(bottom)):
            bottom_data.append(bottom[i].data_tensor.to_cudandarray())
        for i in range(len(bottom)):
            output = self.b[i](*(bottom_data + [tdiff]))
            bottom[i].diff_tensor.add_from_cudandarray(output)

class Reshape(PyLayer):
    def forward(self, bottom, top):
        shape = self.pythonargs
        top[0].reshape(shape)
        top[0].share_data(bottom[0])
        top[0].share_diff(bottom[0])
