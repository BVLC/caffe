function varargout = caffe_safe(varargin)
  try
     switch nargout
	case 0
	 caffe(varargin{:});
	 varargout={};
	case 1
	 varargout{1} = caffe(varargin{:});
	case 2
	 [varargout{1} varargout{2}] = caffe(varargin{:});
  end
  catch
    error('Exception in caffe');
  end
end
