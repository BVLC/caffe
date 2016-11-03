% 
% All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
% 
% All contributions by the University of California:
% Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
% All rights reserved.
% 
% All other contributions:
% Copyright (c) 2014, 2015, the respective contributors
% All rights reserved.
% For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
% 
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
%     * Redistributions of source code must retain the above copyright notice,
%       this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of Intel Corporation nor the names of its contributors
%       may be used to endorse or promote products derived from this software
%       without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
classdef Net < handle
  % Wrapper class of caffe::Net in matlab
  
  properties (Access = private)
    hNet_self
    attributes
    % attribute fields
    %     hLayer_layers
    %     hBlob_blobs
    %     input_blob_indices
    %     output_blob_indices
    %     layer_names
    %     blob_names
  end
  properties (SetAccess = private)
    layer_vec
    blob_vec
    inputs
    outputs
    name2layer_index
    name2blob_index
    layer_names
    blob_names
  end
  
  methods
    function self = Net(varargin)
      % decide whether to construct a net from model_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a net from model_file
        self = caffe.get_net(varargin{:});
        return
      end
      % construct a net from handle
      hNet_net = varargin{1};
      CHECK(is_valid_handle(hNet_net), 'invalid Net handle');
      
      % setup self handle and attributes
      self.hNet_self = hNet_net;
      self.attributes = caffe_('net_get_attr', self.hNet_self);
      
      % setup layer_vec
      self.layer_vec = caffe.Layer.empty();
      for n = 1:length(self.attributes.hLayer_layers)
        self.layer_vec(n) = caffe.Layer(self.attributes.hLayer_layers(n));
      end
      
      % setup blob_vec
      self.blob_vec = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs);
        self.blob_vec(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
      
      % setup input and output blob and their names
      % note: add 1 to indices as matlab is 1-indexed while C++ is 0-indexed
      self.inputs = ...
        self.attributes.blob_names(self.attributes.input_blob_indices + 1);
      self.outputs = ...
        self.attributes.blob_names(self.attributes.output_blob_indices + 1);
      
      % create map objects to map from name to layers and blobs
      self.name2layer_index = containers.Map(self.attributes.layer_names, ...
        1:length(self.attributes.layer_names));
      self.name2blob_index = containers.Map(self.attributes.blob_names, ...
        1:length(self.attributes.blob_names));
      
      % expose layer_names and blob_names for public read access
      self.layer_names = self.attributes.layer_names;
      self.blob_names = self.attributes.blob_names;
    end
    function layer = layers(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer = self.layer_vec(self.name2layer_index(layer_name));
    end
    function blob = blobs(self, blob_name)
      CHECK(ischar(blob_name), 'blob_name must be a string');
      blob = self.blob_vec(self.name2blob_index(blob_name));
    end
    function blob = params(self, layer_name, blob_index)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      CHECK(isscalar(blob_index), 'blob_index must be a scalar');
      blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
    end
    function forward_prefilled(self)
      caffe_('net_forward', self.hNet_self);
    end
    function backward_prefilled(self)
      caffe_('net_backward', self.hNet_self);
    end
    function res = forward(self, input_data)
      CHECK(iscell(input_data), 'input_data must be a cell array');
      CHECK(length(input_data) == length(self.inputs), ...
        'input data cell length must match input blob number');
      % copy data to input blobs
      for n = 1:length(self.inputs)
        self.blobs(self.inputs{n}).set_data(input_data{n});
      end
      self.forward_prefilled();
      % retrieve data from output blobs
      res = cell(length(self.outputs), 1);
      for n = 1:length(self.outputs)
        res{n} = self.blobs(self.outputs{n}).get_data();
      end
    end
    function res = backward(self, output_diff)
      CHECK(iscell(output_diff), 'output_diff must be a cell array');
      CHECK(length(output_diff) == length(self.outputs), ...
        'output diff cell length must match output blob number');
      % copy diff to output blobs
      for n = 1:length(self.outputs)
        self.blobs(self.outputs{n}).set_diff(output_diff{n});
      end
      self.backward_prefilled();
      % retrieve diff from input blobs
      res = cell(length(self.inputs), 1);
      for n = 1:length(self.inputs)
        res{n} = self.blobs(self.inputs{n}).get_diff();
      end
    end
    function copy_from(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      CHECK_FILE_EXIST(weights_file);
      caffe_('net_copy_from', self.hNet_self, weights_file);
    end
    function reshape(self)
      caffe_('net_reshape', self.hNet_self);
    end
    function save(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      caffe_('net_save', self.hNet_self, weights_file);
    end
  end
end
