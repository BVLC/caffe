# Windows Caffe

**This is an experimental, community based branch led by Guillaume Dumont (@willyd). It is a work-in-progress.**

This branch of Caffe ports the framework to Windows.

[![Travis Build Status](https://api.travis-ci.org/BVLC/caffe.svg?branch=windows)](https://travis-ci.org/BVLC/caffe) Travis (Linux build)

[![Build status](https://ci.appveyor.com/api/projects/status/ew7cl2k1qfsnyql4/branch/windows?svg=true)](https://ci.appveyor.com/project/BVLC/caffe/branch/windows) AppVeyor (Windows build)

## Prebuilt binaries

Prebuilt binaries can be downloaded from the latest CI build on appveyor for the following configurations:

- Visual Studio 2015, CPU only, Python 3.5: [Caffe Release](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D3%2C%20WITH_CUDA%3D0), ~~[Caffe Debug](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DDebug%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D3%2C%20WITH_CUDA%3D0)~~

- Visual Studio 2015, CUDA 8.0, Python 3.5: [Caffe Release](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D1%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D3%2C%20WITH_CUDA%3D1)

- Visual Studio 2015, CPU only, Python 2.7: [Caffe Release](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D2%2C%20WITH_CUDA%3D0), [Caffe Debug](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DDebug%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D2%2C%20WITH_CUDA%3D0)

- Visual Studio 2015,CUDA 8.0, Python 2.7: [Caffe Release](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D1%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D2%2C%20WITH_CUDA%3D1)

- Visual Studio 2013, CPU only, Python 2.7: [Caffe Release](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D12%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D2%2C%20WITH_CUDA%3D0), [Caffe Debug](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D12%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DDebug%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D2%2C%20WITH_CUDA%3D0)


## Windows Setup

### Requirements

 - Visual Studio 2013 or 2015
     - Technically only the VS C/C++ compiler is required (cl.exe)
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)

### Optional Dependencies

 - Python for the pycaffe interface. Anaconda Python 2.7 or 3.5 x64 (or Miniconda)
 - Matlab for the matcaffe interface.
 - CUDA 7.5 or 8.0 (use CUDA 8 if using Visual Studio 2015)
 - cuDNN v5

 We assume that `cmake.exe` and `python.exe` are on your `PATH`.

### Configuring and Building Caffe

The fastest method to get started with caffe on Windows is by executing the following commands in a `cmd` prompt (we use `C:\Projects` as a root folder for the remainder of the instructions):
```cmd
C:\Projects> git clone https://github.com/BVLC/caffe.git
C:\Projects> cd caffe
C:\Projects\caffe> git checkout windows
:: Edit any of the options inside build_win.cmd to suit your needs
C:\Projects\caffe> scripts\build_win.cmd
```
The `build_win.cmd` script will download the dependencies, create the Visual Studio project files (or the ninja build files) and build the Release configuration. By default all the required DLLs will be copied (or hard linked when possible) next to the consuming binaries. If you wish to disable this option, you can by changing the command line option `-DCOPY_PREREQUISITES=0`. The prebuilt libraries also provide a `prependpath.bat` batch script that can temporarily modify your `PATH` environment variable to make the required DLLs available.

If you have GCC installed (e.g. through MinGW), then Ninja will detect it before detecting the Visual Studio compiler, causing errors.  In this case you have several options:

- [Pass CMake the path](https://cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F) (Set `CMAKE_C_COMPILER=your/path/to/cl.exe` and `CMAKE_CXX_COMPILER=your/path/to/cl.exe`)
- or Use the Visual Studio Generator by setting `WITH_NINJA` to 0 (This is slower, but may work even if Ninja is failing.)
- or uninstall your copy of GCC 

The path to cl.exe is usually something like 
`"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/your_processor_architecture/cl.exe".`
If you don't want to install Visual Studio, Microsoft's C/C++ compiler [can be obtained here](http://landinghub.visualstudio.com/visual-cpp-build-tools). 

Below is a more complete description of some of the steps involved in building caffe.

### Install the caffe dependencies

By default CMake will download and extract prebuilt dependencies for your compiler and python version. It will create a folder called `libraries` containing all the required dependencies inside your build folder. Alternatively you can build them yourself by following the instructions in the [caffe-builder](https://github.com/willyd/caffe-builder) [README](https://github.com/willyd/caffe-builder/blob/master/README.md).

### Use cuDNN

To use cuDNN the easiest way is to copy the content of the `cuda` folder into your CUDA toolkit installation directory. For example if you installed CUDA 8.0 and downloaded cudnn-8.0-windows10-x64-v5.1.zip you should copy the content of the `cuda` directory to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`. Alternatively, you can define the CUDNN_ROOT cache variable to point to where you unpacked the cuDNN files e.g. `C:/Projects/caffe/cudnn-8.0-windows10-x64-v5.1/cuda`. For example the command in [scripts/build_win.cmd](scripts/build_win.cmd) would become:
```
cmake -G"!CMAKE_GENERATOR!" ^
      -DBLAS=Open ^
      -DCMAKE_BUILD_TYPE:STRING=%CMAKE_CONFIG% ^
      -DBUILD_SHARED_LIBS:BOOL=%CMAKE_BUILD_SHARED_LIBS% ^
      -DBUILD_python:BOOL=%BUILD_PYTHON% ^
      -DBUILD_python_layer:BOOL=%BUILD_PYTHON_LAYER% ^
      -DBUILD_matlab:BOOL=%BUILD_MATLAB% ^
      -DCPU_ONLY:BOOL=%CPU_ONLY% ^
      -DCUDNN_ROOT=C:/Projects/caffe/cudnn-8.0-windows10-x64-v5.1/cuda ^
      -C "%cd%\libraries\caffe-builder-config.cmake" ^
      "%~dp0\.."
```

Alternatively, you can open `cmake-gui.exe` and set the variable from there and click `Generate`.

### Building only for CPU

If CUDA is not installed Caffe will default to a CPU_ONLY build. If you have CUDA installed but want a CPU only build you may use the CMake option `-DCPU_ONLY=1`.

### Using the Python interface

The recommended Python distribution is Anaconda or Miniconda. To successfully build the python interface you need to add the following conda channels:
```
conda config --add channels conda-forge
conda config --add channels willyd
```
and install the following packages:
```
conda install --yes cmake ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz
```
If Python is installed the default is to build the python interface and python layers. If you wish to disable the python layers or the python build use the CMake options `-DBUILD_python_layer=0` and `-DBUILD_python=0` respectively. In order to use the python interface you need to either add the `C:\Projects\caffe\python` folder to your python path or copy the `C:\Projects\caffe\python\caffe` folder to your `site_packages` folder.

### Using the MATLAB interface

Follow the above procedure and use `-DBUILD_matlab=ON`. Change your current directory in MATLAB to `C:\Projects\caffe\matlab` and run the following command to run the tests:
```
>> caffe.run_tests()
```
If all tests pass you can test if the classification_demo works as well. First, from `C:\Projects\caffe` run `python scripts\download_model_binary.py models\bvlc_reference_caffenet` to download the pre-trained caffemodel from the model zoo. Then change your MATLAB directory to `C:\Projects\caffe\matlab\demo` and run `classification_demo`.

### Using the Ninja generator

You can choose to use the Ninja generator instead of Visual Studio for faster builds. To do so, change the option `set WITH_NINJA=1` in the `build_win.cmd` script. To install Ninja you can download the executable from github or install it via conda:
```cmd
> conda config --add channels conda-forge
> conda install ninja --yes
```
When working with ninja you don't have the Visual Studio solutions as ninja is more akin to make. An alternative is to use [Visual Studio Code](https://code.visualstudio.com) with the CMake extensions and C++ extensions.

### Building a shared library

CMake can be used to build a shared library instead of the default static library. To do so follow the above procedure and use `-DBUILD_SHARED_LIBS=ON`. Please note however, that some tests (more specifically the solver related tests) will fail since both the test executable and caffe library do not share static objects contained in the protobuf library.

### Troubleshooting

Should you encounter any error please post the output of the above commands by redirecting the output to a file and open a topic on the [caffe-users list](https://groups.google.com/forum/#!forum/caffe-users) mailing list.

## Known issues

- The `GPUTimer` related test cases always fail on Windows. This seems to be a difference between UNIX and Windows.
- Shared library (DLL) build will have failing tests.
- Shared library build only works with the Ninja generator

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
