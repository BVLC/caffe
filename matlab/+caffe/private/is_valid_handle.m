function valid = is_valid_handle(hObj)
% valid = is_valid_handle(hObj) or is_valid_handle('get_new_init_key')
%   Check if a handle is valid (has the right data type and init_key matches)
%   Use is_valid_handle('get_new_init_key') to get new init_key from C++;

% a handle is a struct array with the following fields
%   (uint64) ptr      : the pointer to the C++ object
%   (double) init_key : caffe initialization key

persistent init_key;
if isempty(init_key)
  init_key = caffe_('get_init_key');
end

% is_valid_handle('get_new_init_key') to get new init_key from C++;
if ischar(hObj) && strcmp(hObj, 'get_new_init_key')
  init_key = caffe_('get_init_key');
  return
else
  % check whether data types are correct and init_key matches
  valid = isstruct(hObj) ...
    && isscalar(hObj.ptr) && isa(hObj.ptr, 'uint64') ...
    && isscalar(hObj.init_key) && isa(hObj.init_key, 'double') ...
    && hObj.init_key == init_key;
end

end
