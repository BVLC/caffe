classdef Blob < handle
  % Wrapper class of caffe::Blob in matlab
  
  properties (Access = private)
    hBlob_self
  end
  
  methods
    function self = Blob(hBlob_blob)
      CHECK(is_valid_handle(hBlob_blob), 'invalid Blob handle');
      
      % setup self handle
      self.hBlob_self = hBlob_blob;
    end
    function shape = shape(self)
      shape = caffe_('blob_get_shape', self.hBlob_self);
    end
    function reshape(self, shape)
      shape = self.check_and_preprocess_shape(shape);
      caffe_('blob_reshape', self.hBlob_self, shape);
    end
    function data = get_data(self)
      data = caffe_('blob_get_data', self.hBlob_self);
    end
    function set_data(self, data)
      data = self.check_and_preprocess_data(data);
      caffe_('blob_set_data', self.hBlob_self, data);
    end
    function diff = get_diff(self)
      diff = caffe_('blob_get_diff', self.hBlob_self);
    end
    function set_diff(self, diff)
      diff = self.check_and_preprocess_data(diff);
      caffe_('blob_set_diff', self.hBlob_self, diff);
    end
  end
  
  methods (Access = private)
    function shape = check_and_preprocess_shape(~, shape)
      CHECK(isempty(shape) || (isnumeric(shape) && isrow(shape)), ...
        'shape must be a integer row vector');
      shape = double(shape);
    end
    function data = check_and_preprocess_data(~, data)
      CHECK(isnumeric(data), 'data or diff must be numeric types');
      if ~isa(data, 'single')
        data = single(data);
      end
    end
  end
end
