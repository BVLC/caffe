function reset()
% reset()
%   reset Caffe to initial status

caffe_('reset');
is_valid_handle('get_new_init_key');

end
