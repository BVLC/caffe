import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 


def save_code_and_label(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    path = params['path']
    np.save(path + "database_code.npy", database_code)
    np.save(path + "database_label.npy", database_labels)
    np.save(path + "validation_code.npy", validation_code)
    np.save(path + "validation_label.npy", validation_labels)


def mean_average_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    
    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
    
    return np.mean(np.array(APx))
        

def get_codes_and_labels(params):
    caffe.set_device(params['gpu_id'])
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    dims = params['image_dims']
    scale = params['scale']
    database = open(params['database'], 'r').readlines()
    validation = open(params['validation'], 'r').readlines()
    batch_size = params['batch_size']

    if 'mean_file' in params:
        mean_file = params['mean_file']
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, mean=np.load(mean_file).mean(1).mean(1), raw_scale=scale)
    else:
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, raw_scale=scale)
    
    database_code = []
    validation_code = []
    database_labels = []
    validation_labels = []
    cur_pos = 0
    
    while 1:
        lines = database[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]
        codes = net.predict(images, oversample=False)
        [database_code.append(c) for c in codes]
        [database_labels.append(l) for l in labels]
        
        print str(cur_pos) + "/" + str(len(database))
        if len(lines) < batch_size:
            break;

    cur_pos = 0
    while 1:
        lines = validation[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]

        codes = net.predict(images, oversample=False)
        [validation_code.append(c) for c in codes]
        [validation_labels.append(l) for l in labels]
        
        print str(cur_pos) + "/" + str(len(validation))
        if len(lines) < batch_size:
            break;
        
    return dict(database_code=database_code, database_labels=database_labels, validation_code=validation_code, validation_labels=validation_labels)


# run 4 threads in each of 3 gpus, total 4*3=12 threads
nthreads = 4
ndevices = 3
params = []

for gpu_id in range(ndevices):
    for i in range(nthreads):
        params.append(dict(model_file="./deploy.prototxt",
                      pretrained_model="../nus_wide/trained_model.caffemodel",
                      image_dims=(256,256),
                      scale=255,
                      database="../../../data/nus_wide/parallel/database" + str(nthreads*gpu_id+i) + ".txt",
                      validation="../../../data/nus_wide/parallel/test" + str(nthreads*gpu_id+i) + ".txt",
                      batch_size=50,
                      mean_file="./ilsvrc_2012_mean.npy",
                      gpu_id=gpu_id))

pool = Pool(nthreads*ndevices)
results = pool.map(get_codes_and_labels, params)

code_and_label = results[0]
for i in range(1, nthreads*ndevices):
    [code_and_label['database_code'].append(c) for c in results[i]['database_code']]
    [code_and_label['database_labels'].append(c) for c in results[i]['database_labels']]
    [code_and_label['validation_code'].append(c) for c in results[i]['validation_code']]
    [code_and_label['validation_labels'].append(c) for c in results[i]['validation_labels']]

code_and_label['R'] = 5000
mAP = mean_average_precision(code_and_label)

aaa = open('./result', 'w')
aaa.write(str(mAP))
print mAP

#code_and_label['path'] = "./data/"
#save_code_and_label(code_and_label)
