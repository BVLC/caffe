"""
List of layer classes for building protobuf layer parameters from python
"""

from .layer_headers import Layer, LossLayer, DataLayer
from .layer_helpers import assign_proto, Filler
from apollocaffe.proto import caffe_pb2

class CapSequence(Layer):
    def __init__(self, name, sequence_lengths, **kwargs):
        super(CapSequence, self).__init__(self, name, kwargs)
        for x in sequence_lengths:
            self.p.rp.cap_sequence_param.sequence_lengths.append(x)

class Concat(Layer):
    def __init__(self, name, **kwargs):
        super(Concat, self).__init__(self, name, kwargs)

class Convolution(Layer):
    def __init__(self, name, kernel_dim, num_output, weight_filler=None, bias_filler=None, **kwargs):
        kwargs['kernel_h'] = kernel_dim[0]
        kwargs['kernel_w'] = kernel_dim[1]
        kwargs['num_output'] = num_output
        super(Convolution, self).__init__(self, name, kwargs)
        if weight_filler is None:
            weight_filler = Filler('xavier')
        self.p.convolution_param.weight_filler.CopyFrom(weight_filler.filler_param)
        if bias_filler is None:
            bias_filler = Filler('constant', 0.)
        self.p.convolution_param.bias_filler.CopyFrom(bias_filler.filler_param)

class Data(DataLayer):
    def __init__(self, name, source, batch_size, transform=None, **kwargs):
        kwargs['source'] = source
        kwargs['batch_size'] = batch_size
        super(Data, self).__init__(self, name, kwargs)
        self.p.data_param.backend = caffe_pb2.DataParameter.LMDB
        if transform is not None:
            self.p.transform_param.CopyFrom(transform.transform_param)

class Deconvolution(Convolution):
    def __init__(self, name, **kwargs):
        kernel_dim = kwargs['kernel_dim']
        num_output = kwargs['num_output']

        del kwargs['kernel_dim']
        del kwargs['num_output']
            
        l = Convolution(name, kernel_dim, num_output, None, None, **kwargs)
        self.p = l.p
        self.p.type = 'Deconvolution'
       

class Dropout(Layer):
    def __init__(self, name, dropout_ratio, **kwargs):
        kwargs['dropout_ratio'] = dropout_ratio
        super(Dropout, self).__init__(self, name, kwargs)

class DummyData(DataLayer):
    def __init__(self, name, shape, **kwargs):
        super(DummyData, self).__init__(self, name, kwargs)
        assert len(shape) == 4
        self.p.dummy_data_param.num.append(shape[0])
        self.p.dummy_data_param.channels.append(shape[1])
        self.p.dummy_data_param.height.append(shape[2])
        self.p.dummy_data_param.width.append(shape[3])

class Eltwise(Layer):
    def __init__(self, name, operation, **kwargs):
        super(Eltwise, self).__init__(self, name, kwargs)
        if operation == 'MAX':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.MAX
        elif operation == 'SUM':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
        elif operation == 'PROD':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.PROD
        else:
            raise ValueError('Unknown Eltwise operator')

class Embed(Layer):
    def __init__(self, name, weight_filler=None, bias_filler=None, **kwargs):
        super(Embed, self).__init__(self, name, kwargs)
        if weight_filler is None:
            weight_filler = Filler('uniform', 0.1)
        self.p.embed_param.weight_filler.CopyFrom(weight_filler.filler_param)
        if bias_filler is not None:
            self.p.embed_param.bias_filler.CopyFrom(bias_filler.filler_param)

class EuclideanLoss(LossLayer):
    def __init__(self, name, **kwargs):
        super(EuclideanLoss, self).__init__(self, name, kwargs)

class HDF5Data(DataLayer):
    def __init__(self, name, source, batch_size, transform=None, **kwargs):
        kwargs['source'] = source
        kwargs['batch_size'] = batch_size
        super(HDF5Data, self).__init__(self, name, kwargs)
        if transform is not None:
            self.p.transform_param.CopyFrom(transform.transform_param)

class ImageData(DataLayer):
    def __init__(self, name, source, batch_size, transform=None, **kwargs):
        kwargs['source'] = source
        kwargs['batch_size'] = batch_size
        super(ImageData, self).__init__(self, name, kwargs)
        if transfrom is not None:
            self.p.transform_param.CopyFrom(transform.transform_param)


class InnerProduct(Layer):
    def __init__(self, name, num_output, weight_filler=None, bias_filler=None, **kwargs):
        kwargs['num_output'] = num_output
        super(InnerProduct, self).__init__(self, name, kwargs)
        if weight_filler is None:
            weight_filler = Filler('xavier')
        self.p.inner_product_param.weight_filler.CopyFrom(weight_filler.filler_param)
        if bias_filler is None:
            bias_filler = Filler('constant', 0.)
        self.p.inner_product_param.bias_filler.CopyFrom(bias_filler.filler_param)

class LRN(Layer):
    def __init__(self, name, **kwargs):
        super(LRN, self).__init__(self, name, kwargs)

class LstmUnit(Layer):
    def __init__(self, name, num_cells, weight_filler=None, **kwargs):
        super(LstmUnit, self).__init__(self, name, kwargs)
        self.p.lstm_unit_param.num_cells = num_cells
        if weight_filler is None:
            weight_filler = Filler('uniform', 0.1)
        self.p.lstm_unit_param.input_weight_filler.CopyFrom(
            weight_filler.filler_param)
        self.p.lstm_unit_param.input_gate_weight_filler.CopyFrom(
            weight_filler.filler_param)
        self.p.lstm_unit_param.forget_gate_weight_filler.CopyFrom(
            weight_filler.filler_param)
        self.p.lstm_unit_param.output_gate_weight_filler.CopyFrom(
            weight_filler.filler_param)

class L1Loss(LossLayer):
    def __init__(self, name, **kwargs):
        super(L1Loss, self).__init__(self, name, kwargs)

class NumpyData(DataLayer):
    def __init__(self, name, data, **kwargs):
        super(NumpyData, self).__init__(self, name, kwargs)
        from apollocaffe import make_numpy_data_param
        import numpy as np
        #self.p.rp.ParseFromString(make_numpy_data_param(np.array(data, dtype=np.float32)).SerializeToString())
        self.p = make_numpy_data_param(self.p, np.array(data, dtype=np.float32))

class Pooling(Layer):
    def __init__(self, name, pool='MAX', **kwargs):
        super(Pooling, self).__init__(self, name, kwargs)
        if pool is not None:
            if pool == 'MAX':
                self.p.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
            elif pool == 'AVE':
                self.p.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
            elif pool == 'STOCHASTIC':
                self.p.pooling_param.pool = caffe_pb2.PoolingParameter.STOCHASTIC
            else:
                raise ValueError('Unknown pooling method')

class Power(Layer):
    def __init__(self, name, **kwargs):
        super(Power, self).__init__(self, name, kwargs)

class ReLU(Layer):
    def __init__(self, name, **kwargs):
        super(ReLU, self).__init__(self, name, kwargs)

class Softmax(Layer):
    def __init__(self, name, **kwargs):
        super(Softmax, self).__init__(self, name, kwargs)

class SoftmaxWithLoss(LossLayer):
    def __init__(self, name, **kwargs):
        super(SoftmaxWithLoss, self).__init__(self, name, kwargs)

class Accuracy(Layer):
    def __init__(self, name, **kwargs):
        super(Accuracy, self).__init__(self, name, kwargs)

class Transpose(Layer):
    def __init__(self, name, **kwargs):
        super(Transpose, self).__init__(self, name, kwargs)

class Unknown(Layer):
    def __init__(self, p):
        self.p = p

class Wordvec(Layer):
    def __init__(self, name, dimension, vocab_size, weight_filler=None, **kwargs):
        kwargs['dimension'] = dimension
        kwargs['vocab_size'] = vocab_size
        super(Wordvec, self).__init__(self, name, kwargs)
        if weight_filler is None:
            weight_filler = Filler('uniform', 0.1)
        self.p.wordvec_param.weight_filler.CopyFrom(weight_filler.filler_param)
