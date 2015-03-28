import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

def conv_relu(bottom, w, b, ks, nout, stride=1, pad=0, group=1, device=None):
    conv = L.Convolution(bottom, w, b, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, group=group, param_bottoms=2, device=device)
    return conv, L.ReLU(conv, in_place=True, device=device)

def fc_relu(bottom, w, b, nout, device=None):
    fc = L.InnerProduct(bottom, w, b, num_output=nout, param_bottoms=2, device=device)
    return fc, L.ReLU(fc, in_place=True, device=device)

def max_pool(bottom, ks, stride=1, device=None):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, device=device)

def param(*shape, **kwargs):
    return L.Parameter(shape=dict(dim=list(shape)), **kwargs)

def alexnet(n_dev):
    n = caffe.NetSpec()
    # params
    n.conv1w = param(96, 3, 11, 11)
    n.conv1b = param(96)
    n.conv2w = param(256, 48, 5, 5)
    n.conv2b = param(256)
    n.conv3w = param(384, 256, 3, 3)
    n.conv3b = param(384)
    n.conv4w = param(384, 192, 3, 3)
    n.conv4b = param(384)
    n.conv5w = param(256, 192, 3, 3)
    n.conv5b = param(256)
    n.fc6w = param(4096, 9216)
    n.fc6b = param(4096)
    n.fc7w = param(4096, 4096)
    n.fc7b = param(4096)
    n.fc8w = param(1000, 4096)
    n.fc8b = param(1000)

    def net(device_id, batch_size):
        device = dict(type=P.Device.GPU, device_id=device_id)
        # dummy data
        data = L.DummyData(shape=[dict(dim=[batch_size, 3, 227, 227])], device=device)
        label = L.DummyData(shape=[dict(dim=[batch_size])], device=device)

        # alexnet
        conv1, relu1 = conv_relu(data, n.conv1w, n.conv1b, 11, 96, stride=4, device=device)
        pool1 = max_pool(relu1, 3, stride=2, device=device)
        norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75, device=device)
        conv2, relu2 = conv_relu(norm1, n.conv2w, n.conv2b, 5, 256, pad=2, group=2, device=device)
        pool2 = max_pool(relu2, 3, stride=2, device=device)
        norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75, device=device)
        conv3, relu3 = conv_relu(norm2, n.conv3w, n.conv3b, 3, 384, pad=1, device=device)
        conv4, relu4 = conv_relu(relu3, n.conv4w, n.conv4b, 3, 384, pad=1, group=2, device=device)
        conv5, relu5 = conv_relu(relu4, n.conv5w, n.conv5b, 3, 256, pad=1, group=2, device=device)
        pool5 = max_pool(relu5, 3, stride=2, device=device)
        fc6, relu6 = fc_relu(pool5, n.fc6w, n.fc6b, 4096, device=device)
        fc7, relu7 = fc_relu(relu6, n.fc7w, n.fc7b, 4096, device=device)
        fc8 = L.InnerProduct(relu7, n.fc8w, n.fc8b, num_output=1000, param_bottoms=2, device=device)
        loss = L.SoftmaxWithLoss(fc8, label, device=device)
        return loss

    for i in range(n_dev):
        setattr(n, 'loss' + str(i + 1), net(i, 256 // n_dev))

    return n.to_proto()
    
def make_net():
    for n in range(1, 4):
        with open('cn-{}x.prototxt'.format(n), 'w') as f:
            print >>f, alexnet(n)

if __name__ == '__main__':
    make_net()
