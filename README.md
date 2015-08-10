# Caffe for Faster R-CNN

Caffe fork that supports Fast**er** R-CNN, forked from [BLVC/caffe](https://github.com/BVLC/caffe) on June 27th, 2015.

### Compile for Windows
0.	Download a VS 2013 solution ([Onedrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17218&authkey=!AOqDbPj7Idd4O4w&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/mqw7b7qqx0dojkb/caffe_library.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1hqGojnI)) which include some related libraries.
0.	Copy all files in this repo to .\caffe in the solution.
0.	Prepare external libraries of OpenCV/Boost/MKL. Refer to the links below:
 - [OpenCV](http://opencv.org/downloads.html)
 - [Boost with pre-bulit binaries](http://sourceforge.net/projects/boost/files/boost-binaries/)
 - [MKL](https://software.intel.com/en-us/intel-parallel-studio-xe)
0.	Switch the configuration to “Release_Mex” for compiling mex for MATLAB interface.
0.	In the VS solution, modify “Include Directories” and “Library Directories” to point to your external libraries.
0.	Set “Caffe” project as startup project.
0.	Rebuild the entire solution.
0.	Copy all files in .\x64\Release_Mex to faster_rcnn-master\external\caffe\matlab\caffe_faster_rcnn.

### Known issues for Windows:
0.	If not using VS 2013, you need to re-build the solution in .\Library\leveldb. Then copy the built leveldb.lib to .\x64\Release_Mex.
0.	If you are not using OpenCV 2.4.9, copy the corresponding opencv dll files to .\x64\Release_Mex.
