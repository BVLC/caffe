classdef test_net < matlab.unittest.TestCase
  
  properties
    num_output
    model_file
    net
  end
  
  methods (Static)
    function model_file = simple_net_file(num_output)
      model_file = tempname();
      fid = fopen(model_file, 'w');
      fprintf(fid, [ ...
        'name: "testnet" force_backward: true\n' ...
        'layer { type: "DummyData" name: "data" top: "data" top: "label"\n' ...
        'dummy_data_param { num: 5 channels: 2 height: 3 width: 4\n' ...
        '    num: 5 channels: 1 height: 1 width: 1\n' ...
        '    data_filler { type: "gaussian" std: 1 }\n' ...
        '    data_filler { type: "constant" } } }\n' ...
        'layer { type: "Convolution" name: "conv" bottom: "data" top: "conv"\n' ...
        '  convolution_param { num_output: 11 kernel_size: 2 pad: 3\n' ...
        '    weight_filler { type: "gaussian" std: 1 }\n' ...
        '    bias_filler { type: "constant" value: 2 } }\n' ...
        '    param { decay_mult: 1 } param { decay_mult: 0 }\n' ...
        '    }\n' ...
        'layer { type: "InnerProduct" name: "ip" bottom: "conv" top: "ip"\n' ...
        '  inner_product_param { num_output: ' num2str(num_output) ...
        '    weight_filler { type: "gaussian" std: 2.5 }\n' ...
        '    bias_filler { type: "constant" value: -3 } } }\n' ...
        'layer { type: "SoftmaxWithLoss" name: "loss" bottom: "ip" bottom: "label"\n' ...
        '  top: "loss" }' ]);
      fclose(fid);
    end
  end
  methods
    function self = test_net()
      self.num_output = 13;
      self.model_file = caffe.test.test_net.simple_net_file(self.num_output);
      self.net = caffe.Net(self.model_file, 'train');
      % also make sure get_solver runs
      caffe.get_net(self.model_file, 'train');
      
      % fill in valid labels
      self.net.blobs('label').set_data(randi( ...
        self.num_output - 1, self.net.blobs('label').shape));
      
      delete(self.model_file);
    end
  end
  methods (Test)
    function test_forward_backward(self)
      self.net.forward_prefilled();
      self.net.backward_prefilled();
    end
    function test_inputs_outputs(self)
      self.verifyEqual(self.net.inputs, cell(0, 1))
      self.verifyEqual(self.net.outputs, {'loss'});
    end
    function test_save_and_read(self)
      weights_file = tempname();
      self.net.save(weights_file);
      model_file2 = caffe.test.test_net.simple_net_file(self.num_output);
      net2 = caffe.Net(model_file2, weights_file, 'train');
      delete(model_file2);
      delete(weights_file);
      for l = 1:length(self.net.layer_vec)
        for i = 1:length(self.net.layer_vec(l).params)
          self.verifyEqual(self.net.layer_vec(l).params(i).get_data(), ...
            net2.layer_vec(l).params(i).get_data());
        end
      end
    end
  end
end