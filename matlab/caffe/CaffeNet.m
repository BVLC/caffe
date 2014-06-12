classdef CaffeNet
  properties
    model_def_file
    model_file
    init_key        
  end
  properties (Hidden)
    cutoff % Defines a cuttoff point
  end
  methods (Access=private)
    function self = CaffeNet(model_def_file, model_file)
      if nargin > 0
        self.model_def_file = model_def_file;
        self.init_key = caffe('init_net', model_def_file);
      end
      if nargin > 1
        self.model_file = model_file;
        self.init_key = caffe('init', model_def_file, model_file);
      end
    end
  end    
  methods (Static)
    function obj = instance(model_def_file, model_file)
      persistent self
      if isempty(self)
        switch nargin
          case 1
            self = CaffeNet(model_def_file);
          case 2
            self = CaffeNet(model_def_file, model_file);
          otherwise
            error('Need to be initialized with at least model_def_file')
        end
      end
      obj = self;
    end
  
  methods
    function res = forward(self,input)
      if nargin < 2
        res = caffe('forward');
      else
        res = caffe('forward',input);
      end
    end
    function res = backward(self,input)
      if nargin < 2
        res = caffe('backward');
      else
        res = caffe('backward',input);
      end
    end
    function res = forward_prefilled(self)
      res = caffe('forward_prefilled');
    end
    function res = backward_prefilled(self)
      res = caffe('backward_prefilled');
    end
    function res = init_net(self, model_def_file)
      self.init_key = caffe('init_net',model_def_file);
      res = self.init_key;
    end
    function res = load_net(self, model_file)
      self.init_key = caffe('load_net',model_file);
      res = self.init_key;
    end
    function res = save_net(self, model_file)
      res = caffe('save_net', model_file);
    end
    function res = is_initialized(self)
      res = caffe('is_initialized');
    end
    function res = set_mode_cpu(self)
      res = caffe('set_mode_cpu');
    end
    function res = set_mode_gpu(self)
      res = caffe('set_mode_gpu');
    end
    function res = set_mode(self,mode)
      % mode = {'cpu' 'gpu'}
      switch mode
        case {'cpu','CPU'}
          res = caffe('set_mode_cpu');
        case {'gpu','GPU'}
          res = caffe('set_mode_gpu');
        otherwise
          error('Mode unknown');
      end
    end
    function res = set_phase_train(self)
      res = caffe('set_phase_train');
    end
    function res = set_phase_test(self)
      res = caffe('set_phase_test');
    end
    function res = set_phase(self, phase)
      % phase = {'train' 'test'}
      switch phase
        case {'train','TRAIN'}
          res = caffe('set_phase_train');
        case {'test','TEST'}
          res = caffe('set_phase_test');
        otherwise
          error('Phase unknown');
      end
    end
    function res = set_device(self, device_id)
      res = caffe('set_device', device_id);
    end
    function res = get_weights(self)
      res = caffe('get_weights');
    end
    function res = set_weights(self, weights)
      res = caffe('set_weights', weights);
    end
    function res = get_layer_weights(self, layer_name)
      res = caffe('get_layer_weights', layer_name);
    end
    function res = set_layer_weights(self, layer_name, weights)
      res = caffe('set_layer_weights', layer_name, weights);
    end
    function res = get_layers_info(self)
      res = caffe('get_layers_info');
    end
    function res = get_blobs_info(self)
      res = caffe('get_blobs_info');
    end
    function res = get_blob_data(self, blob_name)
      res = caffe('get_blob_data', blob_name);
    end
    function res = get_blob_diff(self, blob_name)
      res = caffe('get_blob_diff', blob_name);
    end
    function res = get_all_data(self)
      res = caffe('get_all_data');
    end
    function res = get_all_diff(self)
      res = caffe('get_all_diff');
    end
    function res = get_init_key(self)
      res = caffe('get_init_key');
    end
    function res = reset(self)
      res = caffe('reset');
    end
  end
end
