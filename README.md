# Windows Caffe

**This is an experimental, communtity based branch led by Guillaume Dumont (@willyd). It is a work-in-progress.**

This branch of Caffe ports the framework to Windows.

[![Travis Build Status](https://api.travis-ci.org/BVLC/caffe.svg?branch=windows)](https://travis-ci.org/BVLC/caffe) Travis (Linux build)

[![Windows Build status](https://ci.appveyor.com/api/projects/status/6xpwyq0y9ffdj9pb/branch/windows?svg=true)](https://ci.appveyor.com/project/willyd/caffe-4pvka/branch/windows) AppVeyor (Windows build)

## Windows Setup
**Requirements**:
 - Visual Studio 2013 or 2015
 - CMake 3.4+
 - Python 2.7 Anaconda x64 (or Miniconda)
 - CUDA 7.5 or 8.0 (optional) (use CUDA 8 if using Visual Studio 2015)
 - cuDNN v5 (optional)

you may also like to try the [ninja](https://ninja-build.org/) cmake generator as the build times can be much lower on multi-core machines. ninja can be installed easily with the `conda` package manager by adding the conda-forge channel with:
```cmd
> conda config --add channels conda-forge
> conda install ninja --yes
```
When working with ninja you don't have the Visual Studio solutions as ninja is more akin to make. An alternative is to use [Visual Studio Code](https://code.visualstudio.com) with the CMake extensions and C++ extensions.

### Install the caffe dependencies

The easiest and recommended way of installing the required depedencies is by downloading the pre-built libraries using the `%CAFFE_ROOT%\scripts\download_prebuilt_dependencies.py` file. Depending on your compiler one of the following commands should download and extract the prebuilt dependencies to your current working directory:

```cmd
:: Install Visual Studio 2013 dependencies
> python scripts\download_prebuilt_dependencies.py --msvc_version=v120
:: Or install Visual Studio 2015 dependencies
> python scripts\download_prebuilt_dependencies.py --msvc_version=v140
```

This will create a folder called `libraries` containing all the required dependencies. Alternatively you can build them yourself by following the instructions in the [caffe-builder](https://github.com/willyd/caffe-builder) [README](https://github.com/willyd/caffe-builder/blob/master/README.md). For the remaining of these instructions we will assume that the libraries folder is in a folder defined by the `%CAFFE_DEPENDENCIES%` environment variable.

### Build caffe

If you are using the Ninja generator you need to setup the MSVC compiler using:
```
> call "%VS120COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
```
then from the caffe source folder you need to configure the cmake build
```
> set CMAKE_GENERATOR=Ninja
> set CMAKE_CONFIGURATION=Release
> mkdir build
> cd build
> cmake -G%CMAKE_GENERATOR% -DBLAS=Open -DCMAKE_BUILD_TYPE=%CMAKE_CONFIGURATION% -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<install_path> -C %CAFFE_DEPENDENCIES%\caffe-builder-config.cmake  ..\
> cmake --build . --config %CMAKE_CONFIGURATION%
> cmake --build . --config %CMAKE_CONFIGURATION% --target install
```
In the above command `CMAKE_GENERATOR` can be either `Ninja`, `"Visual Studio 12 2013 Win64"` or `"Visual Studio 14 2015 Win64"` and `CMAKE_CONFIGURATION` can be `Release` or `Debug`. Please note however that Visual Studio will not parallelize the build of the CUDA files which results in much longer build times.

In case one of the steps in the above procedure is not working please refer to the appveyor build scripts in `%CAFFE_ROOT%\scripts\appveyor` to see the most up to date build procedure.

### Use cuDNN

To use cuDNN you need to define the CUDNN_ROOT cache variable to point to where you unpacked the cuDNN files e.g. `C:/Users/myuser/Projects/machine-learning/cudnn-8.0-windows10-x64-v5.1/cuda`. For example, the build command above would become:

```
> set CMAKE_GENERATOR=Ninja
> set CMAKE_CONFIGURATION=Release
> mkdir build
> cd build
> cmake -G%CMAKE_GENERATOR% -DBLAS=Open -DCMAKE_BUILD_TYPE=%CMAKE_CONFIGURATION% -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<install_path> -DCUDNN_ROOT=C:/Users/myuser/Projects/machine-learning/cudnn-8.0-windows10-x64-v5.1/cuda -C %CAFFE_DEPENDENCIES%\caffe-builder-config.cmake  ..\
> cmake --build . --config %CMAKE_CONFIGURATION%
> cmake --build . --config %CMAKE_CONFIGURATION% --target install
```
Make sure to use forward slashes (`/`) in the path. You will need to add the folder containing the cuDNN DLL to your PATH.

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
If Python is installed the default is to build the python interface and python layers. If you wish to disable the python layers or the python build use the CMake options `-DBUILD_python_layer=0` and `-DBUILD_python=0` respectively. In order to use the python interface you need to either add the `%CAFFE_ROOT%\python` folder to your python path of copy the `%CAFFE_ROOT%\python\caffe` folder to your `site_packages` folder. Also, you need to edit your `PATH` or copy the required DLLs next to the `caffe.pyd` file. Only Python 2.7 x64 has been tested on Windows.

### Using the MATLAB interface

Follow the above procedure and use `-DBUILD_matlab=ON`. Then, you need to add the path to the generated `.mexw64` file to your `PATH` and the folder caffe/matlab to your Matlab search PATH to use matcaffe.

### Building a shared library

CMake can be used to build a shared library instead of the default static library. To do so follow the above procedure and use `-DBUILD_SHARED_LIBS=ON`. Please note however, that some tests (more specifically the solver related tests) will fail since both the test exectuable and caffe library do not share static objects contained in the protobuf library.

### Running the tests or the caffe executable

To run the tests or any caffe executable you will have to update your `PATH` to include the directories where the depedencies dlls are located:
```
:: Prepend to avoid conflicts with other libraries with same name
:: For VS 2013
> set PATH=%CAFFE_DEPENDENCIES%\bin;%CAFFE_DEPENDENCIES%\lib;%CAFFE_DEPENDENCIES%\x64\vc12\bin;%PATH%
:: For VS 2015
> set PATH=%CAFFE_DEPENDENCIES%\bin;%CAFFE_DEPENDENCIES%\lib;%CAFFE_DEPENDENCIES%\x64\vc14\bin;%PATH%
```
or you can use the prependpath.bat included with the prebuilt dependencies. Then the tests can be run from the build folder:
```
cmake --build . --target runtest --config %CMAKE_CONFIGURATION%
```

### TODOs
- Python 3.5: Create protobuf packages for 3.5. Rebuild dependencies especially boost python with 3.5.

## Previous Visual Studio based build

The previous windows build based on Visual Studio project files is now deprecated. However, it is still available in the `windows` folder. Please see the [README.md](windows/README.md) in there for details.

## Known issues

- The `GPUTimer` related test cases always fail on Windows. This seems to be a difference between UNIX and Windows.
- Shared library (DLL) build will have failing tests.

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
