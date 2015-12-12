function net = get_net(varargin)
% net = get_net(model_file, phase_name) or
% net = get_net(model_file, weights_file, phase_name)
%   Construct a net from model_file, and load weights from weights_file
%   phase_name can only be 'train' or 'test'

CHECK(nargin == 2 || nargin == 3, ['usage: ' ...
  'net = get_net(model_file, phase_name) or ' ...
  'net = get_net(model_file, weights_file, phase_name)']);
if nargin == 3
  model_file = varargin{1};
  weights_file = varargin{2};
  phase_name = varargin{3};
elseif nargin == 2
  model_file = varargin{1};
  phase_name = varargin{2};
end

CHECK(ischar(model_file), 'model_file must be a string');
CHECK(ischar(phase_name), 'phase_name must be a string');
CHECK_FILE_EXIST(model_file);
CHECK(strcmp(phase_name, 'train') || strcmp(phase_name, 'test'), ...
  sprintf('phase_name can only be %strain%s or %stest%s', ...
  char(39), char(39), char(39), char(39)));

% construct caffe net from model_file
hNet = caffe_('get_net', model_file, phase_name);
net = caffe.Net(hNet);

% load weights from weights_file
if nargin == 3
  CHECK(ischar(weights_file), 'weights_file must be a string');
  CHECK_FILE_EXIST(weights_file);
  net.copy_from(weights_file);
end

end
