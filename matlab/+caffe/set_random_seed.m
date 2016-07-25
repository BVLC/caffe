function set_random_seed(random_seed)
% set_random_seed(random_seed)
%   set Caffe's random_seed

CHECK(isscalar(random_seed) && random_seed >= 0, ...
  'random_seed must be non-negative integer');
random_seed = double(random_seed);

caffe_('set_random_seed', random_seed);

end
