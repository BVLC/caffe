import caffe
import numpy as np
import time
import os
import sys

if len(sys.argv) == 1:
  start_snapshot = 0

nz = 100 # latent vector dimension
image_size = 64 # image size
max_iter = int(1e6) # maximum number of iterations
display_every = 20 # show losses every so many iterations
snapshot_every = 1000 # snapshot every so many iterations
snapshot_folder = 'snapshots_test' # where to save the snapshots (and load from)

feat_shape = (nz,)
im_size = (3,image_size,image_size)
batch_size = 64
snapshot_at_iter = -1
snapshot_at_iter_file = 'snapshot_at_iter.txt'

sub_nets = ('generator', 'discriminator', 'data')

if not os.path.exists(snapshot_folder):
  os.makedirs(snapshot_folder)
    
#make solvers
with open ("solver_template.prototxt", "r") as myfile:
  solver_template=myfile.read()
  
for curr_net in sub_nets:
  with open("solver_%s.prototxt" % curr_net, "w") as myfile:
    myfile.write(solver_template.replace('@NET@', curr_net))                 

#initialize the nets
generator = caffe.AdamSolver('solver_generator.prototxt')
discriminator = caffe.AdamSolver('solver_discriminator.prototxt')
data_reader = caffe.AdamSolver('solver_data.prototxt')


#load from snapshot
if start_snapshot:
  curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
  print >> sys.stderr, '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
  generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
  if os.path.isfile(generator_caffemodel):
    generator.net.copy_from(generator_caffemodel)
  else:
    raise Exception('File %s does not exist' % generator_caffemodel)
  discriminator_caffemodel = curr_snapshot_folder +'/' + 'discriminator.caffemodel'
  if os.path.isfile(discriminator_caffemodel):
    discriminator.net.copy_from(discriminator_caffemodel)
  else:
    raise Exception('File %s does not exist' % discriminator_caffemodel)

#read weights of losses
discr_loss_weight = discriminator.net._blob_loss_weights[discriminator.net._blob_names_index['discr_loss']]

train_discr = True
train_gen = True

#do training
start = time.time()
for it in range(start_snapshot,max_iter):
  # read the data
  data_reader.net.forward_simple()
  # feed the data to the generator and run it
  generator.net.blobs['feat'].data[...] = np.random.normal(0, 1, (64, nz)).astype(np.float32)
  generator.net.forward_simple()
  generated_img = generator.net.blobs['generated'].data
  # run the discriminator on real data
  discriminator.net.blobs['data'].data[...] = data_reader.net.blobs['data'].data
  discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1), dtype='float32')
  discriminator.net.forward_simple()
  discr_real_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_discr:
    discriminator.increment_iter()
    discriminator.net.clear_param_diffs()
    discriminator.net.backward_simple()

  # run the discriminator on generated data
  discriminator.net.blobs['data'].data[...] = generated_img
  discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1), dtype='float32')
  discriminator.net.forward_simple()
  discr_fake_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_discr:
    discriminator.net.backward_simple()
    discriminator.apply_update()

  # run the discriminator on generated data with opposite labels, to get the gradient for the generator
  discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1), dtype='float32')
  discriminator.net.forward_simple()
  discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
  if train_gen:
    generator.increment_iter()
    generator.net.clear_param_diffs()
    discriminator.net.backward_simple()
    
    generator.net.blobs['generated'].diff[...] = discriminator.net.blobs['data'].diff
    generator.net.backward_simple()
    generator.apply_update()

   
  #display
  if it % display_every == 0:
    print >> sys.stderr, "[%s] Iteration %d: %f seconds" % (time.strftime("%c"), it, time.time()-start)
    print >> sys.stderr, "  discr real loss: %e * %e = %f" % (discr_real_loss, discr_loss_weight, discr_real_loss*discr_loss_weight)
    print >> sys.stderr, "  discr fake loss: %e * %e = %f" % (discr_fake_loss, discr_loss_weight, discr_fake_loss*discr_loss_weight)
    print >> sys.stderr, "  discr fake loss for generator: %e * %e = %f" % (discr_fake_for_generator_loss, discr_loss_weight, discr_fake_for_generator_loss*discr_loss_weight)
    start = time.time()
    if os.path.isfile(snapshot_at_iter_file):
      with open (snapshot_at_iter_file, "r") as myfile:
        snapshot_at_iter = int(myfile.read())
    
  #snapshot
  if it % snapshot_every == 0 or it == snapshot_at_iter:
    curr_snapshot_folder = snapshot_folder +'/' + str(it)
    print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
    if not os.path.exists(curr_snapshot_folder):
      os.makedirs(curr_snapshot_folder)
    generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
    generator.net.save(generator_caffemodel)
    discriminator_caffemodel = curr_snapshot_folder + '/' + 'discriminator.caffemodel'
    discriminator.net.save(discriminator_caffemodel)
    
  #switch optimizing discriminator and generator, so that neither of them overfits too much
  discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
  if discr_loss_ratio < 1e-1 and train_discr:    
    train_discr = False
    train_gen = True
    print >> sys.stderr, "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  if discr_loss_ratio > 5e-1 and not train_discr:    
    train_discr = True
    train_gen = True
    print >> sys.stderr, " <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  if discr_loss_ratio > 1e1 and train_gen:
    train_gen = False
    train_discr = True
    print >> sys.stderr, "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen)
  
