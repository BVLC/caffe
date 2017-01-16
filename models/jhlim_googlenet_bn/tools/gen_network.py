filename = 'train_val.body'
f = open(filename, 'w')

def addLinear(bottom, top, num_output): 
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "InnerProduct"\n');
    f.write('  param {\n');
    f.write('    lr_mult: 1\n');
    f.write('    decay_mult: 1\n');
    f.write('  }\n');
    f.write('  inner_product_param {\n');
    f.write('    num_output: {}\n'.format(num_output));
    f.write('    weight_filler {\n');
    f.write('      type: "xavier"\n');
    f.write('    }\n');
    f.write('    bias_term: false\n');
    f.write('  }\n');
    f.write('}\n');
    return top

def addLoss(bottom, top, weight):
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/loss"\n'.format(top));
    f.write('  name: "{}/loss"\n'.format(top));
    f.write('  type: "SoftmaxWithLoss"\n');
    f.write('  loss_weight: {}\n'.format(weight));
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}/prob"\n'.format(top));
    f.write('  name: "{}/prob"\n'.format(top));
    f.write('  type: "Softmax"\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}/prob"\n'.format(top));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/top-1"\n'.format(top));
    f.write('  name: "{}/top-1"\n'.format(top));
    f.write('  type: "Accuracy"\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}/prob"\n'.format(top));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/top-5"\n'.format(top));
    f.write('  name: "{}/top-5"\n'.format(top));
    f.write('  type: "Accuracy"\n');
    f.write('  accuracy_param {\n');
    f.write('    top_k: 5\n');
    f.write('  }\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    return

def addReLU(bottom):
   #bottom = 'inception_5b/pool_proj/bn'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(bottom));
   f.write('  name: "{}/relu"\n'.format(bottom));
   f.write('  type: "ReLU"\n');
   f.write('}\n');
   return bottom 

def addPooling(bottom, top, kernel, stride=1, pad=0, pool='MAX'):
   #bottom = 'conv1/7x7_s2/sc' 
   #top = 'pool1/3x3_s2'
   #pool = 'MAX'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(top));
   f.write('  name: "{}"\n'.format(top));
   f.write('  type: "Pooling"\n');
   f.write('  pooling_param {\n');
   f.write('    pool: {}\n'.format(pool));
   f.write('    kernel_size: {}\n'.format(kernel));
   f.write('    stride: {}\n'.format(stride));
   if pad > 0:
       f.write('    pad: {}\n'.format(pad));
   f.write('  }\n');
   f.write('}\n');
   return top

def addConv(bottom, top, outChannels, kernel, stride=1, pad=0):
   #bottom = 'data'
   #top = 'conv1/7x7_s2'
   f.write('layer {\n');
   f.write(' bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(top));
   f.write('  name: "{}"\n'.format(top));
   f.write('  type: "Convolution"\n');
   f.write('  param {\n');
   f.write('    lr_mult: 1\n');
   f.write('    decay_mult: 1\n');
   f.write('  }\n');
   f.write('  convolution_param {\n');
   f.write('    num_output: {}\n'.format(outChannels));
   f.write('    pad: {}\n'.format(pad));
   f.write('    kernel_size: {}\n'.format(kernel));
   f.write('    stride: {}\n'.format(stride));
   f.write('    weight_filler {\n');
   f.write('      type: "xavier"\n');
   f.write('    }\n');
   f.write('    bias_term: false\n');
   f.write('  }\n');
   f.write('}\n');
   return top 

def addBatchNorm(bottom, top): 
    #bottom = 'conv1/7x7_s2'
    #top = 'conv1/7x7_s2/bn'
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  name: "{}"\n'.format(top));
    f.write('  top: "{}"\n'.format(top));
    f.write('  type: "BatchNorm"\n');
    #f.write('  batch_norm_param {\n');
    #f.write('    use_global_stats: true\n');
    #f.write('  }\n');
    f.write('}\n');
    return top 

def addScale(bottom, top): 
    #bottom = 'conv1/7x7_s2/bn'
    #top = 'conv1/7x7_s2/sc'
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "Scale"\n');
    f.write('  scale_param {\n');
    f.write('    bias_term: true\n');
    f.write('  }\n');
    f.write('}\n');
    return top 

def addConcat(bottoms, top):
    f.write('layer {\n');
    for bottom in bottoms:
        f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "Concat"\n');
    f.write('}\n');
    return top

def addInception(bottom, prefix, num_out_1x1, num_out_3x3_reduce, num_out_3x3, num_out_double3x3_reduce, num_out_double3x3, num_out_pool, pool, stride):
   #3a:  64,     64,  64,     64,  96,     32, 'AVE',   1
   #3b:  64,     64,  96,     64,  96,     64, 'AVE',   1
   #3c:   0,    128, 160,     64,  96,      0, 'MAX',   2

   #4a: 224,     64,  96,     96, 128,    128, 'AVE',   1
   #4b: 192,     96, 128,     96, 128,    128, 'AVE',   1
   #4c: 160,    128, 160,    128, 160,     96, 'AVE',   1
   #4d:  96,    128, 192,    160, 192,     96, 'AVE',   1
   #4e:   0,    128, 192,    192, 256,      0, 'MAX',   2

   #5a: 352,    192, 320,    160, 224,    128, 'AVE',   1
   #5b: 352,    192, 320,    192, 224,    128, 'MAX',   1

   # 1x1
   if num_out_1x1 > 0: 
       top1=addConv(bottom=bottom, top='inception_{}/1x1'.format(prefix), outChannels=num_out_1x1, kernel=1, stride=stride, pad=0)
       top1=addBatchNorm(bottom=top1, top='{}/bn'.format(top1))
       top1=addScale(bottom=top1, top='{}/sc'.format(top1))
       top1=addReLU(bottom=top1)
      
   # 3x3
   top2=addConv(bottom=bottom, top='inception_{}/3x3_reduce'.format(prefix), outChannels=num_out_3x3_reduce, kernel=1, stride=1, pad=0)
   top2=addBatchNorm(bottom=top2, top='{}/bn'.format(top2))
   top2=addScale(bottom=top2, top='{}/sc'.format(top2))
   top2=addReLU(bottom=top2)

   top2=addConv(bottom=top2, top='inception_{}/3x3'.format(prefix), outChannels=num_out_3x3, kernel=3, stride=stride, pad=1)
   top2=addBatchNorm(bottom=top2, top='{}/bn'.format(top2))
   top2=addScale(bottom=top2, top='{}/sc'.format(top2))
   top2=addReLU(bottom=top2)

   # double 3x3
   top3=addConv(bottom=bottom, top='inception_{}/double3x3_reduce'.format(prefix), outChannels=num_out_double3x3_reduce, kernel=1, stride=1, pad=0)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   top3=addConv(bottom=top3, top='inception_{}/double3x3a'.format(prefix), outChannels=num_out_double3x3, kernel=3, stride=1, pad=1)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   top3=addConv(bottom=top3, top='inception_{}/double3x3b'.format(prefix), outChannels=num_out_double3x3, kernel=3, stride=stride, pad=1)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   # pool projection
   if stride == 1:
       top4=addPooling(bottom=bottom, top='inception_{}/pool'.format(prefix), kernel=3, stride=stride, pad=1, pool=pool) 
   elif stride == 2:
       top4=addPooling(bottom=bottom, top='inception_{}/pool'.format(prefix), kernel=3, stride=stride, pad=0, pool=pool) 
   else: 
       raise ValueError('stride is either 1 or 2. in inception layer.') 
   if num_out_pool > 0:
       top4=addConv(bottom=top4, top='inception_{}/pool_proj'.format(prefix), outChannels=num_out_pool, kernel=1, stride=1, pad=0)
       top4=addBatchNorm(bottom=top4, top='{}/bn'.format(top4))
       top4=addScale(bottom=top4, top='{}/sc'.format(top4))
       top4=addReLU(bottom=top4)

   # concat
   tops = [top2, top3, top4]
   if num_out_1x1 > 0:
       tops.insert(0, top1)
   top=addConcat(bottoms=tops, top='inception_{}/output'.format(prefix))

   return top 

def addClassifier(bottom, top, num_output):
    f.write('layer {\n'); 
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "InnerProduct"\n');
    f.write('  param {\n');
    f.write('    lr_mult: 1\n');
    f.write('    decay_mult: 1\n');
    f.write('  }\n');
    f.write('  param {\n');
    f.write('    lr_mult: 2\n');
    f.write('    decay_mult: 0\n');
    f.write('  }\n');
    f.write('  inner_product_param {\n');
    f.write('    num_output: {}\n'.format(num_output));
    f.write('    weight_filler {\n');
    f.write('      type: "xavier"\n');
    f.write('    }\n');
    f.write('    bias_filler {\n');
    f.write('      type: "constant"\n');
    f.write('      value: 0\n');
    f.write('    }\n');
    f.write('  }\n');
    f.write('}\n');
    return top

# inception layers arugments
   #3a:  64,     64,  64,     64,  96,     32, 'AVE',   1
   #3b:  64,     64,  96,     64,  96,     64, 'AVE',   1
   #3c:   0,    128, 160,     64,  96,      0, 'MAX',   2

   #4a: 224,     64,  96,     96, 128,    128, 'AVE',   1
   #4b: 192,     96, 128,     96, 128,    128, 'AVE',   1
   #4c: 160,    128, 160,    128, 160,     96, 'AVE',   1
   #4d:  96,    128, 192,    160, 192,     96, 'AVE',   1
   #4e:   0,    128, 192,    192, 256,      0, 'MAX',   2

   #5a: 352,    192, 320,    160, 224,    128, 'AVE',   1
   #5b: 352,    192, 320,    192, 224,    128, 'MAX',   1

# main
# conv1/7x7_s2
path1=addConv(bottom='data', top='conv1/7x7_s2', outChannels=64, kernel=7, stride=2, pad=3)
path1=addBatchNorm(bottom=path1, top='{}/bn'.format(path1))
path1=addScale(bottom=path1, top='{}/sc'.format(path1))
path1=addReLU(bottom=path1)

# pool1/3x3_s2
path1=addPooling(bottom=path1, top='pool1/3x3_s2', kernel=3, stride=2, pad=0, pool='MAX')

# conv2/3x3_reduce
path1=addConv(bottom=path1, top='conv2/3x3_reduce', outChannels=64, kernel=1, stride=1, pad=0)
path1=addBatchNorm(bottom=path1, top='{}/bn'.format(path1))
path1=addScale(bottom=path1, top='{}/sc'.format(path1))
path1=addReLU(bottom=path1)

# conv2/3x3
path1=addConv(bottom=path1, top='conv2/3x3', outChannels=192, kernel=3, stride=1, pad=1)
path1=addBatchNorm(bottom=path1, top='{}/bn'.format(path1))
path1=addScale(bottom=path1, top='{}/sc'.format(path1))
path1=addReLU(bottom=path1)

# pool2/3x3_s2
path1=addPooling(bottom=path1, top='pool2/3x3_s2', kernel=3, stride=2, pad=0, pool='MAX')

# inception layers
path1=addInception(path1, '3a',  64,     64,  64,     64,  96,     32, 'AVE',   1)
path1=addInception(path1, '3b',  64,     64,  96,     64,  96,     64, 'AVE',   1)
path1=addInception(path1, '3c',   0,    128, 160,     64,  96,      0, 'MAX',   2)

# auxillary classifier1
path2=addPooling(bottom=path1, top='pool3/5x5_s3', kernel=5, stride=3, pad=0, pool='AVE')
path2=addConv(bottom=path2, top='loss1/conv', outChannels=128, kernel=1, stride=1, pad=0)
path2=addBatchNorm(bottom=path2, top='{}/bn'.format(path2))
path2=addScale(bottom=path2, top='{}/sc'.format(path2))
path2=addReLU(bottom=path2)
path2=addLinear(bottom=path2, top='loss1/fc', num_output=1024)
path2=addBatchNorm(bottom=path2, top='{}/bn'.format(path2))
path2=addScale(bottom=path2, top='{}/sc'.format(path2))
path2=addReLU(bottom=path2)
path2=addClassifier(bottom=path2, top='loss1/classifier', num_output=1000)
addLoss(bottom=path2, top='loss1', weight=0.3)

# inception layers
path1=addInception(path1, '4a', 224,     64,  96,     96, 128,    128, 'AVE',   1)
path1=addInception(path1, '4b', 192,     96, 128,     96, 128,    128, 'AVE',   1)
path1=addInception(path1, '4c', 160,    128, 160,    128, 160,     96, 'AVE',   1)
path1=addInception(path1, '4d',  96,    128, 192,    160, 192,     96, 'AVE',   1)
path1=addInception(path1, '4e',   0,    128, 192,    192, 256,      0, 'MAX',   2)

# auxillary classifier2
path3=addPooling(bottom=path1, top='pool4/5x5_s3', kernel=5, stride=3, pad=0, pool='AVE')
path3=addConv(bottom=path3, top='loss2/conv', outChannels=128, kernel=1, stride=1, pad=0)
path3=addBatchNorm(bottom=path3, top='{}/bn'.format(path3))
path3=addScale(bottom=path3, top='{}/sc'.format(path3))
path3=addReLU(bottom=path3)
path3=addLinear(bottom=path3, top='loss2/fc', num_output=1024)
path3=addBatchNorm(bottom=path3, top='{}/bn'.format(path3))
path3=addScale(bottom=path3, top='{}/sc'.format(path3))
path3=addReLU(bottom=path3)
path3=addClassifier(bottom=path3, top='loss2/classifier', num_output=1000)
addLoss(bottom=path3, top='loss2', weight=0.3)

# inception layers
path1=addInception(path1, '5a', 352,    192, 320,    160, 224,    128, 'AVE',   1)
path1=addInception(path1, '5b', 352,    192, 320,    192, 224,    128, 'MAX',   1)

# main classifier
path1=addPooling(bottom=path1, top='pool5/7x7_s1', kernel=7, stride=1, pad=0, pool='AVE')
path1=addClassifier(bottom=path1, top='loss3/classifier', num_output=1000)
addLoss(bottom=path1, top='loss3', weight=1)


