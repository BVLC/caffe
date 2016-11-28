# Windows Caffe

**This is an experimental, communtity based branch led by Guillaume Dumont (@willyd). It is a work-in-progress.**

This branch of Caffe ports the framework to Windows.

[![Travis Build Status](https://api.travis-ci.org/BVLC/caffe.svg?branch=windows)](https://travis-ci.org/BVLC/caffe) Travis (Linux build)

[![Windows Build status](https://ci.appveyor.com/api/projects/status/6xpwyq0y9ffdj9pb/branch/windows?svg=true)](https://ci.appveyor.com/project/willyd/caffe-4pvka/branch/windows) AppVeyor (Windows build)

## Windows Setup

### Requirements

 - Visual Studio 2013 or 2015
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)
 - Python 2.7 Anaconda x64 (or Miniconda).
 - CUDA 7.5 or 8.0 (optional) (use CUDA 8 if using Visual Studio 2015)
 - cuDNN v5 (optional)

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
The `build_win.cmd` script should be executed once to download the dependencies, create the Visual Studio project files (or the ninja build files) and build the Release configuration. After that you should add the required folders to your `PATH` by executing the following command:
```cmd
C:\Projects\caffe> call build\libraries\prependpath.bat
```
Once this is done you can use the `pycaffe` interface or run `caffe.exe` from the command line. If you want to debug the `caffe.exe` exectuable, open Visual Studio from a `cmd.exe` prompt that has the required directories in its `PATH` variable and open the `C:\Projects\caffe\build\Caffe.sln` and proceed as normal. Alternatively, you can copy the required DLLs next to the `caffe.exe` ( or `caffe-d.exe` in Debug).

Should you encounter any error please post the output of the above commands by redirecting the output to a file and open a topic on the [caffe-users list](https://groups.google.com/forum/#!forum/caffe-users) mailing list.

Below is a more complete description of some of the steps involved in building caffe.

### Install the caffe dependencies

The easiest and recommended way of installing the required dependencies is by downloading the pre-built libraries using the [scripts\download_prebuilt_dependencies.py](scripts\download_prebuilt_dependencies.py) file. Depending on your compiler one of the following commands should download and extract the prebuilt dependencies to your current working directory:

```cmd
:: Install Visual Studio 2013 dependencies
> python scripts\download_prebuilt_dependencies.py --msvc_version=v120
:: Or install Visual Studio 2015 dependencies
> python scripts\download_prebuilt_dependencies.py --msvc_version=v140
```

This will create a folder called `libraries` containing all the required dependencies. Alternatively you can build them yourself by following the instructions in the [caffe-builder](https://github.com/willyd/caffe-builder) [README](https://github.com/willyd/caffe-builder/blob/master/README.md). For the remaining of these instructions we will assume that the libraries folder is in a folder defined by the `%CAFFE_DEPENDENCIES%` environment variable.

### Use cuDNN

To use cuDNN you need to define the CUDNN_ROOT cache variable to point to where you unpacked the cuDNN files e.g. `C:/Projects/caffe/cudnn-8.0-windows10-x64-v5.1/cuda`. For example the command in [scripts/build_win.cmd](scripts/build_win.cmd) would become:
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

The recommended Python distribution is Anaconda or Miniconda. To successfully build the python interface you need to install the following packages:
```
conda install --yes numpy scipy matplotlib scikit-image pip six
```
also you will need a protobuf python package that is compatible with pre-built dependencies. This package can be installed this way:
```
conda config --add channels willyd
conda install --yes protobuf==3.1.0.vc12
```
If Python is installed the default is to build the python interface and python layers. If you wish to disable the python layers or the python build use the CMake options `-DBUILD_python_layer=0` and `-DBUILD_python=0` respectively. In order to use the python interface you need to either add the `C:\Projects\caffe\python` folder to your python path of copy the `C:\Projects\caffe\python\caffe` folder to your `site_packages` folder. Also, you need to edit your `PATH` or copy the required DLLs next to the `caffe.pyd` file. Only Python 2.7 x64 has been tested on Windows.

### Using the MATLAB interface

TODO


### Using the Ninja generator

You can choose to use the Ninja generator instead of Visual Studio for faster builds. To do so, change the option `set WITH_NINJA=1` in the `build_win.cmd` script. To install Ninja you can download the executable from github or install it via conda:
```cmd
> conda config --add channels conda-forge
> conda install ninja --yes
```
When working with ninja you don't have the Visual Studio solutions as ninja is more akin to make. An alternative is to use [Visual Studio Code](https://code.visualstudio.com) with the CMake extensions and C++ extensions.

### Building a shared library

CMake can be used to build a shared library instead of the default static library. To do so follow the above procedure and use `-DBUILD_SHARED_LIBS=ON`. Please note however, that some tests (more specifically the solver related tests) will fail since both the test exectuable and caffe library do not share static objects contained in the protobuf library.

### TODOs
- Python 3.5: Create protobuf packages for 3.5. Rebuild dependencies especially boost python with 3.5.

## Previous Visual Studio based build

The previous windows build based on Visual Studio project files is now deprecated. However, it is still available in the `windows` folder. Please see the [README.md](windows/README.md) in there for details.

## Known issues

- The `GPUTimer` related test cases always fail on Windows. This seems to be a difference between UNIX and Windows.
- Shared library (DLL) build will have failing tests.

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
