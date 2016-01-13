import os
import sys
from pylab import *
import random
import numpy as np

caffe_root_dir=os.path.dirname(os.path.realpath(__file__))
caffe_root_dir+='/../../python'
sys.path.insert(0, caffe_root_dir)
if len (sys.argv) < 4:
    raise RuntimeError('Usage: python ' + sys.argv[0] + ' path_to_solver path_to_save_model mode')
import caffe

solver_path = str(sys.argv[1])
init_path = str(sys.argv[2])
init_mode =  str(sys.argv[3])
margin = 0.02;
max_iter = 20;

mode_check=False;  
if init_mode == 'Orthonormal':
    mode_check=True
elif init_mode == 'LSUV':
    mode_check=True
elif init_mode == 'OrthonormalLSUV':
    mode_check=True
else:
    raise RuntimeError('Unknown mode. Try Orthonormal or LSUV or  OrthonormalLSUV')

caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_path)
if os.path.isfile(init_path):
    print "Loading"
    solver.net.copy_from(init_path)

def svd_orthonormal(shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q

for k,v in solver.net.params.iteritems():
    try:
        print k, v[0].data.shape,v[1].data.shape
    except:
        print 'Error with layer',k, 'skipping it'
        continue
    if ('BN' in k) or ('bn' in k):
        print 'Skipping BatchNorm layer'
        continue;
    if 'Orthonormal' in init_mode:
        weights=svd_orthonormal(v[0].data[:].shape)
        solver.net.params[k][0].data[:]=weights#* sqrt(2.0/(1.0+neg_slope*neg_slope));
    else:
        weights=solver.net.params[k][0].data[:]
    if 'LSUV' in init_mode:
        solver.net.forward()
        v = solver.net.blobs[k];
        var1=np.var(v.data[:]);
        mean1 = np.mean(v.data[:]);
        print k,'var = ', var1,'mean = ', mean1
        sys.stdout.flush()
        iter_num = 0;
        while (abs(1.0 - var1) > margin):
            weights=solver.net.params[k][0].data[:]
            solver.net.params[k][0].data[:] = weights / sqrt(var1);
            solver.net.forward()
            v = solver.net.blobs[k];
            var1=np.var(v.data[:]);
            mean1 = np.mean(v.data[:]);
            print k,'var = ', var1,'mean = ', mean1
            sys.stdout.flush()
            iter_num+=1;
            if iter_num > max_iter:
                print 'Could not converge in ', iter_num, ' iterations, go to next layer'
                break; 
print "Initialization finished!"
for k,v in solver.net.params.iteritems():
    vv = solver.net.blobs[k];
    try:
        print k,vv.data[:].shape, ' var = ', np.var(vv.data[:]), ' mean = ', np.mean(vv.data[:])
    except:
        print 'Cannot proceed layer',k,'skiping'
        
print "Saving model..."
solver.net.save(init_path)
print "Finished. Model saved to", init_path