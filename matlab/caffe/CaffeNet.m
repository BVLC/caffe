classdef CaffeNet < handle
    properties
        model_def_file = '../../examples/imagenet/imagenet_deploy.prototxt';
        model_file = '../../examples/imagenet/caffe_reference_imagenet_model';
        mode
        phase
        device_id
        weights
        layers_info
        blobs_info
    end
    properties (Hidden)
        init_key
        weights_changed = true;
        getting_weights = false;
    end
    methods (Access=private)
        function self = CaffeNet(model_def_file, model_file)
            if nargin > 0
                self.model_def_file = model_def_file;
                self.init_key = caffe('init_net', model_def_file);
                self.layers_info = caffe('get_layers_info');
                self.blobs_info = caffe('get_blobs_info');
            end
            if nargin > 1
                self.model_file = model_file;
                self.init_key = caffe('init', model_def_file, model_file);
                self.weights_changed  = true;
            end
        end
    end
    methods (Static)
        function obj = instance(model_def_file, model_file)
            persistent self
            if nargin < 1 || isempty(model_def_file)
                % By default use imagenet_deploy
                model_def_file = '../../examples/imagenet/imagenet_deploy.prototxt';
            end
            if nargin < 2 || isempty(model_file)
                % By default use caffe reference model
                model_file = '../../examples/imagenet/caffe_reference_imagenet_model';
            end
            if isempty(self)
                self = CaffeNet(model_def_file, model_file);
            end
            obj = self;
        end
    end
    methods
        function weights = get.weights(self)
            if (weights_changed)
                self.getting_weights = true;
                self.weights = caffe('get_weights');
                self.getting_weights = false;
                self.weights_changed = false;
            else
                weights = self.weights;
            end
        end
        function set.weights(self,weights)
            if (self.getting_weights)
                self.weights = weights;
            else
                caffe('set_weights', weights);
                self.weights = weights;
                self.weights_changed = false;
            end
        end
        function set.mode(self,mode)
            % mode = {'CPU' 'GPU'}
            switch mode
                case 'CPU'
                    caffe('set_mode_cpu');
                    self.mode = mode;
                case 'GPU'
                    caffe('set_mode_gpu');
                    self.mode = mode;
                otherwise
                    fprintf('Mode unknown choose between CPU and GPU\n');
                    error('Mode unknown');
            end
        end
        function set.phase(self, phase)
            % phase = {'TRAIN' 'TEST'}
            switch phase
                case 'TRAIN'
                    caffe('set_phase_train');
                    self.phase = phase;
                case {'test','TEST'}
                    caffe('set_phase_test');
                    self.phase = phase;
                otherwise
                    fprintf('Phase unknown choose between TRAIN and TEST')
                    error('Phase unknown');
            end
        end
        function set.device_id(self, device_id)
            caffe('set_device', device_id);
            self.device_id = device_id;
        end
    end
    methods
        function res = forward(~,input)
            if nargin < 2
                res = caffe('forward');
            else
                res = caffe('forward',input);
            end
        end
        function res = backward(~,diff)
            if nargin < 2
                res = caffe('backward');
            else
                res = caffe('backward',diff);
            end
        end
        function res = forward_prefilled(~)
            res = caffe('forward_prefilled');
        end
        function res = backward_prefilled(~)
            res = caffe('backward_prefilled');
        end
        function res = init_net(self, model_def_file)
            self.init_key = caffe('init_net',model_def_file);
            self.layers_info = caffe('get_layers_info');
            self.blobs_info = caffe('get_blobs_info');
            self.weights_changed = true;
            self.model_file = null;
            res = self.init_key;
        end
        function res = load_net(self, model_file)
            self.init_key = caffe('load_net',model_file);
            self.model_file = model_file;
            self.weights_changed = true;
            res = self.init_key;
        end
        function res = save_net(~, model_file)
            res = caffe('save_net', model_file);
        end
        function res = is_initialized(~)
            res = caffe('is_initialized');
        end
        function set_mode_cpu(self)
            self.mode = 'CPU';
        end
        function set_mode_gpu(self)
            self.mode = 'GPU';
        end
        function set_phase_train(self)
            self.phase = 'TRAIN';
        end
        function set_phase_test(self)
            self.phase = 'TEST';
        end
        function set_device(self, device_id)
            self.device_id = device_id;
        end
        function res = get_weights(self)
            res = self.weights;
        end
        function res = set_weights(self, weights)
            self.weights = weights;
        end
        function res = get_layer_weights(~, layer_name)
            res = caffe('get_layer_weights', layer_name);
        end
        function res = set_layer_weights(self, layer_name, weights)
            res = caffe('set_layer_weights', layer_name, weights);
            self.weights = caffe('get_weights');
        end
        function res = get_layers_info(self)
            res = self.layers_info;
        end
        function res = get_blobs_info(self)
            res = self.blobs_info;
        end
        function res = get_blob_data(~, blob_name)
            res = caffe('get_blob_data', blob_name);
        end
        function res = get_blob_diff(~, blob_name)
            res = caffe('get_blob_diff', blob_name);
        end
        function res = get_all_data(~)
            res = caffe('get_all_data');
        end
        function res = get_all_diff(~)
            res = caffe('get_all_diff');
        end
        function res = get_init_key(self)
            res = caffe('get_init_key');
            assert(res==self.init_key);
        end
        function res = reset(self)
            res = caffe('reset');
            self.init_key = caffe('get_init_key');
        end
    end
end
