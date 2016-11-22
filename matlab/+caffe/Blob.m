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
    function copy_data_from(self, blob)
      caffe_('blob_copy_data', self.hBlob_self, blob.hBlob_self);  
    end
  end
  
  methods (Access = private)
    function shape = check_and_preprocess_shape(~, shape)
      CHECK(isempty(shape) || (isnumeric(shape) && isrow(shape)), ...
        'shape must be a integer row vector');
      shape = double(shape);
    end
    function data = check_and_preprocess_data(self, data)
      CHECK(isnumeric(data), 'data or diff must be numeric types');
      self.check_data_size_matches(data);
      if ~isa(data, 'single')
        data = single(data);
      end
    end
    function check_data_size_matches(self, data)
      % check whether size of data matches shape of this blob
      % note: matlab arrays always have at least 2 dimensions. To compare
      % shape between size of data and shape of this blob, extend shape of
      % this blob to have at least 2 dimensions
      self_shape_extended = self.shape;
      if isempty(self_shape_extended)
        % target blob is a scalar (0 dim)
        self_shape_extended = [1, 1];
      elseif isscalar(self_shape_extended)
        % target blob is a vector (1 dim)
        self_shape_extended = [self_shape_extended, 1];
      end
      % Also, matlab cannot have tailing dimension 1 for ndim > 2, so you
      % cannot create 20 x 10 x 1 x 1 array in matlab as it becomes 20 x 10
      % Extend matlab arrays to have tailing dimension 1 during shape match
      data_size_extended = ...
        [size(data), ones(1, length(self_shape_extended) - ndims(data))];
      is_matched = ...
        (length(self_shape_extended) == length(data_size_extended)) ...
        && all(self_shape_extended == data_size_extended);
      CHECK(is_matched, ...
        sprintf('%s, input data/diff size: [ %s] vs target blob shape: [ %s]', ...
        'input data/diff size does not match target blob shape', ...
        sprintf('%d ', data_size_extended), sprintf('%d ', self_shape_extended)));
    end
  end
end
