classdef test_io < matlab.unittest.TestCase
  methods (Test)
    function test_read_write_mean(self)
      % randomly generate mean data
      width = 200;
      height = 300;
      channels = 3;
      mean_data_write = 255 * rand(width, height, channels, 'single');
      % write mean data to binary proto
      mean_proto_file = tempname();
      caffe.io.write_mean(mean_data_write, mean_proto_file);
      % read mean data from saved binary proto and test whether they are equal
      mean_data_read = caffe.io.read_mean(mean_proto_file);
      self.verifyEqual(mean_data_write, mean_data_read)
      delete(mean_proto_file);
    end
  end
end
