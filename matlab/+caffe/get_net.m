% 
% All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
% 
% All contributions by the University of California:
% Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
% All rights reserved.
% 
% All other contributions:
% Copyright (c) 2014, 2015, the respective contributors
% All rights reserved.
% For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
% 
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
%     * Redistributions of source code must retain the above copyright notice,
%       this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of Intel Corporation nor the names of its contributors
%       may be used to endorse or promote products derived from this software
%       without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
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
