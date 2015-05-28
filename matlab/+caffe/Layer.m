classdef Layer < handle
  % Wrapper class of caffe::Layer in matlab
  
  properties (Access = private)
    hLayer_self
    attributes
    % attributes fields:
    %     hBlob_blobs
  end
  properties (SetAccess = private)
    params
  end
  
  methods
    function self = Layer(hLayer_layer)
      CHECK(is_valid_handle(hLayer_layer), 'invalid Layer handle');
      
      % setup self handle and attributes
      self.hLayer_self = hLayer_layer;
      self.attributes = caffe_('layer_get_attr', self.hLayer_self);
      
      % setup weights
      self.params = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs)
        self.params(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
    end
    function layer_type = type(self)
      layer_type = caffe_('layer_get_type', self.hLayer_self);
    end
  end
end
