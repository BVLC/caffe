[ -f xor_data.hdf5 ] && echo "Found xor_data.hdf5 file" || ./create_hdf5_data.py
h5dump xor_data.hdf5

# They only data file we use for training here:
echo 'xor_data.hdf5' > xor_data.filelist.txt

# solver.prototxt links to network xor_net.prototxt
# xor_net.prototxt specifies training filelist xor_data.filelist.txt
# TODO training files should NOT be part of the NET definition!!

caffe train -solver=solver.prototxt