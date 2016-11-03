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
%% WRITING TO HDF5
filename='trial.h5';

num_total_samples=10000;
% to simulate data being read from disk / generated etc.
data_disk=rand(5,5,1,num_total_samples); 
label_disk=rand(10,num_total_samples); 

chunksz=100;
created_flag=false;
totalct=0;
for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  last_read=(batchno-1)*chunksz;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=data_disk(:,:,1,last_read+1:last_read+chunksz); 
  batchlabs=label_disk(:,last_read+1:last_read+chunksz);

  % store to hdf5
  startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
  curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end

% display structure of the stored HDF5 file
h5disp(filename);

%% READING FROM HDF5

% Read data and labels for samples #1000 to 1999
data_rd=h5read(filename, '/data', [1 1 1 1000], [5, 5, 1, 1000]);
label_rd=h5read(filename, '/label', [1 1000], [10, 1000]);
fprintf('Testing ...\n');
try 
  assert(isequal(data_rd, single(data_disk(:,:,:,1000:1999))), 'Data do not match');
  assert(isequal(label_rd, single(label_disk(:,1000:1999))), 'Labels do not match');

  fprintf('Success!\n');
catch err
  fprintf('Test failed ...\n');
  getReport(err)
end

%delete(filename);

% CREATE list.txt containing filename, to be used as source for HDF5_DATA_LAYER
FILE=fopen('list.txt', 'w');
fprintf(FILE, '%s', filename);
fclose(FILE);
fprintf('HDF5 filename listed in %s \n', 'list.txt');

% NOTE: In net definition prototxt, use list.txt as input to HDF5_DATA as: 
% layer {
%   name: "data"
%   type: "HDF5Data"
%   top: "data"
%   top: "labelvec"
%   hdf5_data_param {
%     source: "/path/to/list.txt"
%     batch_size: 64
%   }
% }
