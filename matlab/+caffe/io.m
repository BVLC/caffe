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
classdef io
  % a class for input and output functions
  
  methods (Static)
    function im_data = load_image(im_file)
      % im_data = load_image(im_file)
      %   load an image from disk into Caffe-supported data format
      %   switch channels from RGB to BGR, make width the fastest dimension
      %   and convert to single
      %   returns im_data in W x H x C. For colored images, C = 3 in BGR
      %   channels, and for grayscale images, C = 1
      CHECK(ischar(im_file), 'im_file must be a string');
      CHECK_FILE_EXIST(im_file);
      im_data = imread(im_file);
      % permute channels from RGB to BGR for colored images
      if size(im_data, 3) == 3
        im_data = im_data(:, :, [3, 2, 1]);
      end
      % flip width and height to make width the fastest dimension
      im_data = permute(im_data, [2, 1, 3]);
      % convert from uint8 to single
      im_data = single(im_data);
    end
    function mean_data = read_mean(mean_proto_file)
      % mean_data = read_mean(mean_proto_file)
      %   read image mean data from binaryproto file
      %   returns mean_data in W x H x C with BGR channels
      CHECK(ischar(mean_proto_file), 'mean_proto_file must be a string');
      CHECK_FILE_EXIST(mean_proto_file);
      mean_data = caffe_('read_mean', mean_proto_file);
    end
    function write_mean(mean_data, mean_proto_file)
      % write_mean(mean_data, mean_proto_file)
      %   write image mean data to binaryproto file
      %   mean_data should be W x H x C with BGR channels
      CHECK(ischar(mean_proto_file), 'mean_proto_file must be a string');
      CHECK(isa(mean_data, 'single'), 'mean_data must be a SINGLE matrix');
      caffe_('write_mean', mean_data, mean_proto_file);
    end   
  end
end
