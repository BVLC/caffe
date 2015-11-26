To use clBLAS tuned GEMV/GEMM KDB file for a specific GPU, 
rename one of files in this folder to "GPU device name.kdb". 

For example, run a commnand below.
    Go to Caffe root
    cd kdb 
    cp Intel\(R\)\ HD\ Graphics.kdb.Brixbox Intel\(R\)\ HD\ Graphics.kdb

Set CLBLAS_STORAGE_PATH to the current folder. 
    export CLBLAS_STORAGE_PATH=<your current folder, (i.e., $CAFFE_ROOT\kdb)>
By setting CLBLAS_STORAGE_PATH, clBLAS will use automatically *.kdb file. 

Note that each kdb file is per GPU device. If your GPU device is different from 
any of pre-created KDB files, you should not set CLBLAS_STORAGE_PATH. 
    Intel(R) HD Graphics.kdb.Brixbox:   Created on a Brixbox system with Intel Iris Pro GPU (GT3e)
    



