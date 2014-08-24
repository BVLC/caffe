---
layout: default
---
# Solver Optimization

The solver orchestrates model optimization by coordinating the network's forward inference and backward gradients to form parameter updates that attempt to improve the loss.
The responsibilities of learning are divided between the solver for overseeing the optimization and generating parameter updates and the network for yielding loss and gradients.

The Caffe solvers are Stochastic Gradient Descent (SGD), Adaptive Gradient (ADAGRAD), and Nesterov's Accelerated Gradient (NAG).

The solver

1. scaffolds the optimization bookkeeping and creates the training network for learning and test network(s) for evaluation.
2. iteratively optimizes by calling forward / backward and updating parameters
3. (periodically) evaluates the test networks
4. snapshots the model and solver state throughout the optimization

where each iteration

1. calls network forward to make the output and loss
2. calls network backward to make the gradients
3. incorporates the gradients into parameter updates according to the solver method
4. updates the solver state according to learning rate, history, and method

to take the weights all the way from initialization to learned model.

Like Caffe models, Caffe solvers run in CPU / GPU modes.

## Methods

The solver methods address the general optimization problem of loss minimization.
The optimization objective is the average loss over instances

$L(W) = \frac{1}{N} \sum_i f_W^{(i)}\left(X^{(i)}\right) + \lambda r(W)$

where $f_W^{(i)}$ is the loss on instance $i$ with data $X^{(i)}$ in a mini-batch of size $N$ and $r(W)$ is a regularization term with weight $\lambda$.

The model computes $f_W$ in the forward pass and the gradient $\nabla f_W$ in the backward pass.

The gradient of the loss $\nabla L(W)$ is formed by the solver from the model gradient $\nabla f_W$, the regularlization gradient $r(W)$, and other particulars to each method.
The method then computes the parameter update $\Delta W$ to update the weights and iterate.

### SGD

Stochastic gradient descent (SGD)

TODO Bottou pointer

### ADAGRAD

The adaptive gradient (ADAGRAD)

TODO cite Duchi

### NAG

Nesterov's accelerated gradient (NAG)

TODO cite ???

## Scaffolding

The solver scaffolding prepares the optimization method and initializes the model to be learned in `Solver::Presolve()`.

    > caffe train -solver examples/mnist/lenet_solver.prototxt
    I0902 13:35:56.474978 16020 caffe.cpp:90] Starting Optimization
    I0902 13:35:56.475190 16020 solver.cpp:32] Initializing solver from parameters:
    test_iter: 100
    test_interval: 500
    base_lr: 0.01
    display: 100
    max_iter: 10000
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    momentum: 0.9
    weight_decay: 0.0005
    snapshot: 5000
    snapshot_prefix: "examples/mnist/lenet"
    solver_mode: GPU
    net: "examples/mnist/lenet_train_test.prototxt"

Net initialization

    I0902 13:35:56.655681 16020 solver.cpp:72] Creating training net from net file: examples/mnist/lenet_train_test.prototxt
    [...]
    I0902 13:35:56.656740 16020 net.cpp:56] Memory required for data: 0
    I0902 13:35:56.656791 16020 net.cpp:67] Creating Layer mnist
    I0902 13:35:56.656811 16020 net.cpp:356] mnist -> data
    I0902 13:35:56.656846 16020 net.cpp:356] mnist -> label
    I0902 13:35:56.656874 16020 net.cpp:96] Setting up mnist
    I0902 13:35:56.694052 16020 data_layer.cpp:135] Opening lmdb examples/mnist/mnist_train_lmdb
    I0902 13:35:56.701062 16020 data_layer.cpp:195] output data size: 64,1,28,28
    I0902 13:35:56.701146 16020 data_layer.cpp:236] Initializing prefetch
    I0902 13:35:56.701196 16020 data_layer.cpp:238] Prefetch initialized.
    I0902 13:35:56.701212 16020 net.cpp:103] Top shape: 64 1 28 28 (50176)
    I0902 13:35:56.701230 16020 net.cpp:103] Top shape: 64 1 1 1 (64)
    [...]
    I0902 13:35:56.703737 16020 net.cpp:67] Creating Layer ip1
    I0902 13:35:56.703753 16020 net.cpp:394] ip1 <- pool2
    I0902 13:35:56.703778 16020 net.cpp:356] ip1 -> ip1
    I0902 13:35:56.703797 16020 net.cpp:96] Setting up ip1
    I0902 13:35:56.728127 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
    I0902 13:35:56.728142 16020 net.cpp:113] Memory required for data: 5039360
    I0902 13:35:56.728175 16020 net.cpp:67] Creating Layer relu1
    I0902 13:35:56.728194 16020 net.cpp:394] relu1 <- ip1
    I0902 13:35:56.728219 16020 net.cpp:345] relu1 -> ip1 (in-place)
    I0902 13:35:56.728240 16020 net.cpp:96] Setting up relu1
    I0902 13:35:56.728256 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
    I0902 13:35:56.728270 16020 net.cpp:113] Memory required for data: 5167360
    I0902 13:35:56.728287 16020 net.cpp:67] Creating Layer ip2
    I0902 13:35:56.728304 16020 net.cpp:394] ip2 <- ip1
    I0902 13:35:56.728333 16020 net.cpp:356] ip2 -> ip2
    I0902 13:35:56.728356 16020 net.cpp:96] Setting up ip2
    I0902 13:35:56.728690 16020 net.cpp:103] Top shape: 64 10 1 1 (640)
    I0902 13:35:56.728705 16020 net.cpp:113] Memory required for data: 5169920
    I0902 13:35:56.728734 16020 net.cpp:67] Creating Layer loss
    I0902 13:35:56.728747 16020 net.cpp:394] loss <- ip2
    I0902 13:35:56.728767 16020 net.cpp:394] loss <- label
    I0902 13:35:56.728786 16020 net.cpp:356] loss -> loss
    I0902 13:35:56.728811 16020 net.cpp:96] Setting up loss
    I0902 13:35:56.728837 16020 net.cpp:103] Top shape: 1 1 1 1 (1)
    I0902 13:35:56.728849 16020 net.cpp:109]     with loss weight 1
    I0902 13:35:56.728878 16020 net.cpp:113] Memory required for data: 5169924

Loss

    I0902 13:35:56.728893 16020 net.cpp:170] loss needs backward computation.
    I0902 13:35:56.728909 16020 net.cpp:170] ip2 needs backward computation.
    I0902 13:35:56.728924 16020 net.cpp:170] relu1 needs backward computation.
    I0902 13:35:56.728938 16020 net.cpp:170] ip1 needs backward computation.
    I0902 13:35:56.728953 16020 net.cpp:170] pool2 needs backward computation.
    I0902 13:35:56.728970 16020 net.cpp:170] conv2 needs backward computation.
    I0902 13:35:56.728984 16020 net.cpp:170] pool1 needs backward computation.
    I0902 13:35:56.728998 16020 net.cpp:170] conv1 needs backward computation.
    I0902 13:35:56.729014 16020 net.cpp:172] mnist does not need backward computation.
    I0902 13:35:56.729027 16020 net.cpp:208] This network produces output loss
    I0902 13:35:56.729053 16020 net.cpp:467] Collecting Learning Rate and Weight Decay.
    I0902 13:35:56.729071 16020 net.cpp:219] Network initialization done.
    I0902 13:35:56.729085 16020 net.cpp:220] Memory required for data: 5169924
    I0902 13:35:56.729277 16020 solver.cpp:156] Creating test net (#0) specified by net file: examples/mnist/lenet_train_test.prototxt

Completion

    I0902 13:35:56.806970 16020 solver.cpp:46] Solver scaffolding done.
    I0902 13:35:56.806984 16020 solver.cpp:165] Solving LeNet


## Updating Parameters

The actual weight update is made by the solver then applied to the net parameters in `Solver::ComputeUpdateValue()`.

TODO

## Snapshotting and Resuming

The solver snapshots the weights and its own state during training in `Solver::Snapshot()` and `Solver::SnapshotSolverState()`.
The weight snapshots export the learned model while the solver snapshots allow training to be resumed from a given point.
Training is resumed by `Solver::Restore()` and `Solver::RestoreSolverState()`.

Weights are saved without extension while solver states are saved with `.solverstate` extension.
Both files will have an `_iter_N` suffix for the snapshot iteration number.

Snapshotting is configured by:

    # The snapshot interval in iterations.
    snapshot: 5000
    # File path prefix for snapshotting model weights and solver state.
    # Note: this is relative to the invocation of the `caffe` utility, not the
    # solver definition file.
    snapshot_prefix: /path/to/model
    # Snapshot the diff along with the weights. This can help debugging training
    # but takes more storage.
    snapshot_diff: false
    # A final snapshot is saved at the end of training unless
    # this flag is set to false. The default is true.
    snapshot_after_train: true

in the solver definition prototxt.
