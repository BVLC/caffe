#ALL_SHARE=-sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 

#mkdir -p ./log && bsub -b -I -J sw_gan -q q_sw_yfb -host_stack 1024 -N 1 -cgsp 64 -sw3run ../../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ../../bin/caffe_gan train --d_solver=solver_discriminator.prototxt --g_solver=solver_generator.prototxt 2>&1 | tee ./log/sw_gan_mpi_8.log

build/tools/caffe_gan train \
    --d_solver=examples/mnist_gan/d_solver_mnist.prototxt \
    --g_solver=examples/mnist_gan/g_solver_mnist.prototxt