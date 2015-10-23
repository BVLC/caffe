__author__ = 'pittnuts'
import caffe

caffe_root ='./'

#pretrained_net = caffe.Net(caffe_root + 'models/eilab_reference_sparsenet/train_val_scnn.prototxt',
#                           caffe_root + 'models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel',
#                           caffe.TRAIN)
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(caffe_root + 'models/eilab_reference_sparsenet/solver.prototxt')
solver.net.copy_from('models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel')
solver.solve()
