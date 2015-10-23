__author__ = 'wew57'
'''
analyze the weights of caffemodel
'''
from numpy import *
import matplotlib.pyplot as plt
from scipy.io import *

from PIL import Image
#import sys
import caffe
from pittnuts import *
import re
from decimal import *
# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the net, list its data and params, and filter an example image.
# caffe.set_mode_cpu()
# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('../models/bvlc_reference_caffenet/deploy.prototxt','../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
#net = caffe.Net('models/eilab_reference_sparsenet/deploy_scnn.prototxt','models/eilab_reference_sparsenet/sparsenet_train_iter_1000.caffemodel', caffe.TEST)
net = caffe.Net('models/eilab_reference_sparsenet/deploy_scnn.prototxt','models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel', caffe.TEST)
#net = caffe.Net('../examples/cifar10/cifar10_full_scnn.prototxt','../examples/cifar10/eilab_cifar10_full_ini_sparsenet.caffemodel', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
coefficient = 0.00005
group_weight_decay = 0.00001
sum_l1 = 0
sum_l2 = 0
for idx in range(0,len(net.params.keys())):
    key = net.params.keys()[idx]
    if re.match("^conv[1-9]$",key) or re.match("^fc[1-9]$",key):
        print '{}:'.format(key)
        weights = net.params[key][0].data
        sum_l2 += sum(weights**2)*0.5*coefficient
        sum_l1 +=  sum(abs(weights))*coefficient
        #print "mean: {}".format(abs(weights).mean())
        #print "sum of abs {}".format(sum(abs(weights)))
        group_lasso_term = 0
        print "l2:{} l1:{}".format(sum(weights**2)*0.5*coefficient,sum(abs(weights))*coefficient)
        if re.match("^conv[1-9]$",key):
            temp = zeros((1,weights.shape[1]))
            temp_diff = zeros((weights.shape[0],weights.shape[1]))
            for c in range(0,weights.shape[1]):
                    tmp=weights[:,c,:,:].flatten()
                    tmp=dot(tmp,tmp)
                    temp[0,c]= sqrt(tmp)
                    group_lasso_term += temp[0,c]

            for n in range(0,weights.shape[0]):
                for c in range(0,weights.shape[1]):
                    if temp[0,c]==0:
                        temp_diff[n,c]=0
                    else:
                        temp_diff[n,c]=weights[n,c,0,0]/temp[0,c]


        print "group_lasso_term = {}".format(group_lasso_term*group_weight_decay)

        #print "sum of temp: {}".format(sum(abs(temp.flatten()))*96)
        #print "sum of temp: {}".format(sum(abs(temp_diff.flatten())))
        print "asum of diff : {}".format(sum(abs(group_weight_decay*temp_diff.flatten())))

print "\n\nl2 regularization: {}\nl1 regularization: {}".format(sum_l2,sum_l1)






#for idx in range(0,5):
#    data = net.params[net.params.keys()[idx]][0].data
#    #if idx != 0:
#    #    data = data[0:data.shape[0]/2,:,:,:]
#    weights = transpose(data,(2,3,1,0))
#    P,S,Q,q = kernel_factorization(weights)
#    print weights.shape
#    #print q
#    weights_recover,R = kernel_recover(P,S,Q,q)
#    print "[{}] W: %{} error".format(net.params.keys()[idx],100*sum((weights-weights_recover)**2)/sum((weights)**2))
#    r_width = 0.0001
#    print "[{}] S: %{} zeros".format(net.params.keys()[idx],100*sum((abs(S)<r_width).flatten())/(float)(S.size))
#    print "[{}] R: %{} zeros".format(net.params.keys()[idx],100*sum((abs(R)<r_width).flatten())/(float)(R.size))
#
#for key_idx in range(0,len(net.blobs.keys())):
#    print net.blobs.keys()[key_idx], ' -> ', net.blobs[net.blobs.keys()[key_idx]].data.shape
#    #print net.blobs[net.blobs.keys()[key_idx]].data.shape


## display bias and weights of conv1
## print net.params['conv1'][1].data
#weights = transpose(net.params['conv1'][0].data)
##savemat('./conv1weights.mat',{'data':weights})

#
##plt.hist(log10(abs(R)).flatten(),bins=100)
##plt.hist(log10(abs(weights)).flatten(),bins=100)
#plt.hist(S.flatten(),bins=1000,range=(-r_width,r_width))
#print sum((abs(S)<r_width).flatten())
#print S.size
##print abs(S).max(),abs(S).min()
##plt.hist(weights.flatten(),bins=1000)
##print sum(weights)
##print sum(weights_recover)
##h = histogram(S,bins=1000,range=(-r_width,r_width))
##print h



#plt.show()