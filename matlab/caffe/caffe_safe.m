function vargout = caffe_safe(vargin)
  try
  	vargout = caffe(vargin);
  catch
  	error('Exception in caffe');
  end
 end