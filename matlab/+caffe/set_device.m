function set_device(device_id)
% set_device(device_id)
%   set Caffe's GPU device ID

CHECK(isscalar(device_id) && device_id >= 0, ...
  'device_id must be non-negative integer');
device_id = double(device_id);

caffe_('set_device', device_id);

end
