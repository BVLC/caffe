name: "MNISTAutoencoder"
layer {
  name: "data"
  type: "Data"
  top: "data"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "examples/mnist/mnist_train_TRAIN_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  include {
    phase: TEST
    stage: "test-on-train"
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "examples/mnist/mnist_train_TESTonTRAIN_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  include {
    phase: TEST
    stage: "test-on-test"
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "flatdata"
  type: "Flatten"
  bottom: "data"
  top: "flatdata"
}
layer {
  name: "encode1"
  type: "InnerProduct"
  bottom: "data"
  top: "encode1"
  param {
    name: "w1"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b1e"
    lr_mult: 1
    decay_mult: 0
  }

  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "encode1neuron"
  type: "Sigmoid"
  bottom: "encode1"
  top: "encode1neuron"
}
layer {
  name: "encode2"
  type: "InnerProduct"
  bottom: "encode1neuron"
  top: "encode2"
  param {
    name: "w2"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b2e"
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "encode2neuron"
  type: "Sigmoid"
  bottom: "encode2"
  top: "encode2neuron"
}
layer {
  name: "encode3"
  type: "InnerProduct"
  bottom: "encode2neuron"
  top: "encode3"
  param {
    name: "w3"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b3e"
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 250
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "encode3neuron"
  type: "Sigmoid"
  bottom: "encode3"
  top: "encode3neuron"
}
layer {
  name: "encode4"
  type: "InnerProduct"
  bottom: "encode3neuron"
  top: "encode4"
  param {
    name: "w4"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b4e"
    lr_mult: 1
    decay_mult: 0
    
  }
  inner_product_param {
    num_output: 30
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "decode4"
  type: "InnerProduct"
  bottom: "encode4"
  top: "decode4"
  param {
    name: "w4"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b4d"
    lr_mult: 1
    decay_mult: 0
    
  }
  inner_product_param {
    transpose: true
    num_output: 250
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "decode4neuron"
  type: "Sigmoid"
  bottom: "decode4"
  top: "decode4neuron"
}
layer {
  name: "decode3"
  type: "InnerProduct"
  bottom: "decode4neuron"
  top: "decode3"
  param {
    name: "w3"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b3d"
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 500
    transpose: true
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "decode3neuron"
  type: "Sigmoid"
  bottom: "decode3"
  top: "decode3neuron"
}
layer {
  name: "decode2"
  type: "InnerProduct"
  bottom: "decode3neuron"
  top: "decode2"
  param {
    name: "w2"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b2d"
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    transpose: true
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "decode2neuron"
  type: "Sigmoid"
  bottom: "decode2"
  top: "decode2neuron"
}
layer {
  name: "decode1"
  type: "InnerProduct"
  bottom: "decode2neuron"
  top: "decode1"
  param {
    name: "w1"
    lr_mult: 1
    decay_mult: 1
    share_mode: STRICT
  }
  param {
    name: "b1d"
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 784
    transpose: true
    weight_filler {
      type: "gaussian"
      std: 1
      sparse: 15
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss"
  type: "SigmoidCrossEntropyLoss"
  bottom: "decode1"
  bottom: "flatdata"
  top: "cross_entropy_loss"
  loss_weight: 1
}
layer {
  name: "decode1neuron"
  type: "Sigmoid"
  bottom: "decode1"
  top: "decode1neuron"
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "decode1neuron"
  bottom: "flatdata"
  top: "l2_error"
  loss_weight: 0
}
