__author__ = 'pittnuts'

from numpy import *
from numpy.linalg import inv
from scipy.linalg import eigh
import warnings
# PCA analysis
def pca(X):
    """
    :param X: n-by-d data matrix X. Rows of X correspond to observations and columns correspond to dimensions.
    :return: Y - transformed new coordinate in new space, eig_vec - eigenvectors, eig_value
    """
    #X = array(X);
    if (X.ndim != 2) or (X.shape[0]==1):
        raise ValueError('dimension of X should be 2 and X shoud have >1 samples')

    #mean
    avg = mean(X,axis=0)
    shift = abs(avg/std(X,axis=0))
    if shift.mean()>0.1:
        warnings.warn("shift of mean/std: max=={:f}, mean=={:f}".format(shift.max(), shift.mean()),RuntimeWarning)
        pass
        #print shift.max(), shift.mean()

    #deprecated
    #avg = tile(avg,(X.shape[0],1))
    ##X -= avg; #don't change the input value
    #C = dot((X-avg).transpose(),X-avg)/(X.shape[0]-1)

    #covariance matrix
    C = dot(X.transpose(),X)/(X.shape[0]-1)
    C = (C+C.transpose())/2 #make sure it is more symmetric

    #get eigenvalues and eigenvectors
    eig_values,eig_vecs = linalg.eig(C)
    idx = eig_values.argsort()
    idx = idx[ : :-1]
    eig_values = eig_values[idx]
    eig_vecs = eig_vecs[:,idx]

    #new coordinate in new space
    Y = dot(X,eig_vecs)
    
    return (Y, eig_vecs, eig_values)


#refer to paper "Boyuan Liu, sparse convolutional neural networks"
def kernel_factorization(K):
    """
    :param K: a 4D kernel/filer of convolutional layer (filter size x filter size x channel # x filter #)
    :return: factors
    """
    #if want to get a sparse R, mean value of R should be reduced
    # ??????????
    
    #K = array(K)
    filter_size = K.shape[0]
    channel = K.shape[2]
    filter_num = K.shape[3]

    #reshape 4D K to 2D along channel dimension
    K2D = transpose(K,(0,1,3,2)).reshape(((filter_size**2)*filter_num,channel))
    assert (K == K2D.reshape(filter_size,filter_size,filter_num,channel).transpose((0,1,3,2))).all()

    #PCA: mean value is NOT intrinsically reduced
    R2D, P, eig_values_K = pca(K2D)
    assert (R2D == dot(K2D,P)).all()
    P = transpose(P)

    #print inv(P)
    #print inv(P)
    # #assert (transpose(P) == inv(P)).all()

    R = R2D.reshape(filter_size,filter_size,filter_num,channel).transpose((0,1,3,2))
    #print "R: %{} zeros".format(100*sum((abs(R)<0.0001).flatten())/(float)(R.size))
    #check
    for u in range(0,filter_size):
        for v in range(0,filter_size):
            for i in range(0,channel):
                for j in range(0,filter_num):
                    assert abs(K[u,v,i,j] - dot(R[u,v,:,j],P[:,i])) < 0.000001

    # initilize factors for all channels
    S = zeros((channel,filter_size**2,filter_num))
    Q = zeros((channel,filter_size,filter_size,filter_size**2))
    q = zeros(channel)

    for ch in range(0,channel):
        tmp = R[:,:,ch,:].transpose().reshape((filter_num,filter_size**2))
        S_tmp, Q_tmp, eig_values_tmp = pca(tmp)
        #get the index of eigenvalues with 99% accumulative sum
        accumu_rate = cumsum(eig_values_tmp)/sum(eig_values_tmp)
        q[ch] = nonzero(accumu_rate>0.99)[0][0] #the first index with 99% accumulative sum
        #print eig_values_tmp
        #print accumu_rate

        assert (dot(tmp,Q_tmp)==S_tmp).all()

        #get factors
        #print transpose(Q_tmp)
        #print inv(Q_tmp)
        S[ch] = S_tmp.transpose() #.reshape(filter_num,filter_size,filter_size).transpose(2,1,0)
        Q_tmp = Q_tmp.transpose()
        Q[ch] = Q_tmp.reshape(filter_size**2,filter_size,filter_size).transpose()

    #check
    count = 0;
    for u in range(0,filter_size):
        for v in range(0,filter_size):
            for i in range(0,channel):
                for j in range(0,filter_num):
                    assert abs(R[u,v,i,j] - dot(S[i,:,j],Q[i,u,v,:]))<0.000001
                    if abs(R[u,v,i,j] - dot(S[i,:,j],Q[i,u,v,:]))>0.0000001:
                        count += 1
    #print count

    return (P,S,Q,q)

#recover kernel from factors returned from kernel_factorization
def kernel_recover(P,S,Q,q):
    """
    :param P: bases for K
    :param S: coefficient of Q
    :param Q: bases for R
    :param q: number of preserved dimension
    :return: original and new kernels
    """
    filter_size = Q.shape[1]
    channel = P.shape[1]
    filter_num = S.shape[2]
    R = zeros((filter_size,filter_size,channel,filter_num))
    K = zeros((filter_size,filter_size,channel,filter_num))

    #recover R
    for i in range(0,channel):
        #S_mean = mean(S[i],1)
        #S_shift = S[i] - tile(S_mean.reshape((S[i].shape[0],1)),(1,S[i].shape[1]))
        for u in range(0,filter_size):
            for v in range(0,filter_size):
                    for j in range(0,filter_num):
                        #R[u,v,i,j] = dot(S_shift[0:q[i],j],Q[i,u,v,0:q[i]]) + dot(S_mean,Q[i,u,v,:])
                        #R[u,v,i,j] = dot(S_shift[:,j],Q[i,u,v,:]) + dot(S_mean,Q[i,u,v,:])
                        R[u,v,i,j] = dot(S[i,0:q[i],j],Q[i,u,v,0:q[i]])

    #recover K
    for u in range(0,filter_size):
        for v in range(0,filter_size):
            for i in range(0,channel):
                for j in range(0,filter_num):
                    K[u,v,i,j] = dot(R[u,v,:,j],P[:,i])

    return (K,R)

def contains(X,v):
    X = array(X).flatten()
    for x in X:
        if x==v:
            return True

    return False

def contains2D(X,v):
    X = array(X)
    row_num = X.shape[0]
    res = zeros((row_num,1))
    for i in range(0,row_num):
        if contains(X[i,:],v[i]):
            res[i,0] = True
        else: res[i,0] = False
    return res

def zero_out(X,thre):
    orig_shape = X.shape
    zero_out_idx = nonzero(abs(X)<thre)
    #print X[zero_out_idx]
    X[zero_out_idx] = 0
    #assert (X[zero_out_idx] == 0).all()

def zerout_smallest(X,thre):
    orig_shape = X.shape
    zero_out_idx = nonzero(abs(X)<thre)
    #print X[zero_out_idx]
    X[zero_out_idx] = 0
    #assert (X[zero_out_idx] == 0).all()

def get_sparsity(X,thre=0):
    #flatten_X = X.flatten()
    thre = abs(thre)
    mask_mat = abs(X)<=thre
    return sum(mask_mat)/(float)(mask_mat.size)
