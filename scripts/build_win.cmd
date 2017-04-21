@echo off
@setlocal EnableDelayedExpansion

:: Default values
if DEFINED APPVEYOR (
    echo Setting Appveyor defaults
    if NOT DEFINED MSVC_VERSION set MSVC_VERSION=14
    if NOT DEFINED WITH_NINJA set WITH_NINJA=1
    if NOT DEFINED CPU_ONLY set CPU_ONLY=1
    if NOT DEFINED CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto
    if NOT DEFINED CMAKE_CONFIG set CMAKE_CONFIG=Release
    if NOT DEFINED USE_NCCL set USE_NCCL=0
    if NOT DEFINED CMAKE_BUILD_SHARED_LIBS set CMAKE_BUILD_SHARED_LIBS=0
    if NOT DEFINED PYTHON_VERSION set PYTHON_VERSION=2
    if NOT DEFINED BUILD_PYTHON set BUILD_PYTHON=1
    if NOT DEFINED BUILD_PYTHON_LAYER set BUILD_PYTHON_LAYER=1
    if NOT DEFINED BUILD_MATLAB set BUILD_MATLAB=0
    if NOT DEFINED PYTHON_EXE set PYTHON_EXE=python
    if NOT DEFINED RUN_TESTS set RUN_TESTS=1
    if NOT DEFINED RUN_LINT set RUN_LINT=1
    if NOT DEFINED RUN_INSTALL set RUN_INSTALL=1

    :: Set python 2.7 with conda as the default python
    if !PYTHON_VERSION! EQU 2 (
        set CONDA_ROOT=C:\Miniconda-x64
    )
    :: Set python 3.5 with conda as the default python
    if !PYTHON_VERSION! EQU 3 (
        set CONDA_ROOT=C:\Miniconda35-x64
    )
    set PATH=!CONDA_ROOT!;!CONDA_ROOT!\Scripts;!CONDA_ROOT!\Library\bin;!PATH!

    :: Check that we have the right python version
    !PYTHON_EXE! --version
    :: Add the required channels
    conda config --add channels conda-forge
    conda config --add channels willyd
    :: Update conda
    conda update conda -y
    :: Download other required packages
    conda install --yes cmake ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz

    if ERRORLEVEL 1  (
      echo ERROR: Conda update or install failed
      exit /b 1
    )

    :: Install cuda and disable tests if needed
    if !WITH_CUDA! == 1 (
        call %~dp0\appveyor\appveyor_install_cuda.cmd
        set CPU_ONLY=0
        set RUN_TESTS=0
        set USE_NCCL=1
    ) else (
        set CPU_ONLY=1
    )

    :: Disable the tests in debug config
    if "%CMAKE_CONFIG%" == "Debug" (
        echo Disabling tests on appveyor with config == %CMAKE_CONFIG%
        set RUN_TESTS=0
    )

    :: Disable linting with python 3 until we find why the script fails
    if !PYTHON_VERSION! EQU 3 (
        set RUN_LINT=0
    )

) else (
    :: Change the settings here to match your setup
    :: Change MSVC_VERSION to 12 to use VS 2013
    if NOT DEFINED MSVC_VERSION set MSVC_VERSION=14
    :: Change to 1 to use Ninja generator (builds much faster)
    if NOT DEFINED WITH_NINJA set WITH_NINJA=1
    :: Change to 1 to build caffe without CUDA support
    if NOT DEFINED CPU_ONLY set CPU_ONLY=0
    :: Change to generate CUDA code for one of the following GPU architectures
    :: [Fermi  Kepler  Maxwell  Pascal  All]
    if NOT DEFINED CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto
    :: Change to Debug to build Debug. This is only relevant for the Ninja generator the Visual Studio generator will generate both Debug and Release configs
    if NOT DEFINED CMAKE_CONFIG set CMAKE_CONFIG=Release
    :: Set to 1 to use NCCL
    if NOT DEFINED USE_NCCL set USE_NCCL=0
    :: Change to 1 to build a caffe.dll
    if NOT DEFINED CMAKE_BUILD_SHARED_LIBS set CMAKE_BUILD_SHARED_LIBS=0
    :: Change to 3 if using python 3.5 (only 2.7 and 3.5 are supported)
    if NOT DEFINED PYTHON_VERSION set PYTHON_VERSION=2
    :: Change these options for your needs.
    if NOT DEFINED BUILD_PYTHON set BUILD_PYTHON=1
    if NOT DEFINED BUILD_PYTHON_LAYER set BUILD_PYTHON_LAYER=1
    if NOT DEFINED BUILD_MATLAB set BUILD_MATLAB=0
    :: If python is on your path leave this alone
    if NOT DEFINED PYTHON_EXE set PYTHON_EXE=python
    :: Run the tests
    if NOT DEFINED RUN_TESTS set RUN_TESTS=0
    :: Run lint
    if NOT DEFINED RUN_LINT set RUN_LINT=0
    :: Build the install target
    if NOT DEFINED RUN_INSTALL set RUN_INSTALL=0
)

:: Set the appropriate CMake generator
:: Use the exclamation mark ! below to delay the
:: expansion of CMAKE_GENERATOR
if %WITH_NINJA% EQU 0 (
    if "%MSVC_VERSION%"=="14" (
        set CMAKE_GENERATOR=Visual Studio 14 2015 Win64
    )
    if "%MSVC_VERSION%"=="12" (
        set CMAKE_GENERATOR=Visual Studio 12 2013 Win64
    )
    if "!CMAKE_GENERATOR!"=="" (
        echo ERROR: Unsupported MSVC version
        exit /B 1
    )
) else (
    set CMAKE_GENERATOR=Ninja
)

echo INFO: ============================================================
echo INFO: Summary:
echo INFO: ============================================================
echo INFO: MSVC_VERSION               = !MSVC_VERSION!
echo INFO: WITH_NINJA                 = !WITH_NINJA!
echo INFO: CMAKE_GENERATOR            = "!CMAKE_GENERATOR!"
echo INFO: CPU_ONLY                   = !CPU_ONLY!
echo INFO: CUDA_ARCH_NAME             = !CUDA_ARCH_NAME!
echo INFO: CMAKE_CONFIG               = !CMAKE_CONFIG!
echo INFO: USE_NCCL                   = !USE_NCCL!
echo INFO: CMAKE_BUILD_SHARED_LIBS    = !CMAKE_BUILD_SHARED_LIBS!
echo INFO: PYTHON_VERSION             = !PYTHON_VERSION!
echo INFO: BUILD_PYTHON               = !BUILD_PYTHON!
echo INFO: BUILD_PYTHON_LAYER         = !BUILD_PYTHON_LAYER!
echo INFO: BUILD_MATLAB               = !BUILD_MATLAB!
echo INFO: PYTHON_EXE                 = "!PYTHON_EXE!"
echo INFO: RUN_TESTS                  = !RUN_TESTS!
echo INFO: RUN_LINT                   = !RUN_LINT!
echo INFO: RUN_INSTALL                = !RUN_INSTALL!
echo INFO: ============================================================

:: Build and exectute the tests
:: Do not run the tests with shared library
if !RUN_TESTS! EQU 1 (
    if %CMAKE_BUILD_SHARED_LIBS% EQU 1 (
        echo WARNING: Disabling tests with shared library build
        set RUN_TESTS=0
    )
)

if NOT EXIST build mkdir build
pushd build

:: Setup the environement for VS x64
set batch_file=!VS%MSVC_VERSION%0COMNTOOLS!..\..\VC\vcvarsall.bat
call "%batch_file%" amd64

:: Configure using cmake and using the caffe-builder dependencies
:: Add -DCUDNN_ROOT=C:/Projects/caffe/cudnn-8.0-windows10-x64-v5.1/cuda ^
:: below to use cuDNN
cmake -G"!CMAKE_GENERATOR!" ^
      -DBLAS=Open ^
      -DCMAKE_BUILD_TYPE:STRING=%CMAKE_CONFIG% ^
      -DBUILD_SHARED_LIBS:BOOL=%CMAKE_BUILD_SHARED_LIBS% ^
      -DBUILD_python:BOOL=%BUILD_PYTHON% ^
      -DBUILD_python_layer:BOOL=%BUILD_PYTHON_LAYER% ^
      -DBUILD_matlab:BOOL=%BUILD_MATLAB% ^
      -DCPU_ONLY:BOOL=%CPU_ONLY% ^
      -DCOPY_PREREQUISITES:BOOL=1 ^
      -DINSTALL_PREREQUISITES:BOOL=1 ^
      -DUSE_NCCL:BOOL=!USE_NCCL! ^
      -DCUDA_ARCH_NAME:STRING=%CUDA_ARCH_NAME% ^
      "%~dp0\.."

if ERRORLEVEL 1 (
  echo ERROR: Configure failed
  exit /b 1
)

:: Lint
if %RUN_LINT% EQU 1 (
    cmake --build . --target lint  --config %CMAKE_CONFIG%
)

if ERRORLEVEL 1 (
  echo ERROR: Lint failed
  exit /b 1
)

:: Build the library and tools
cmake --build . --config %CMAKE_CONFIG%

if ERRORLEVEL 1 (
  echo ERROR: Build failed
  exit /b 1
)

:: Build and exectute the tests
if !RUN_TESTS! EQU 1 (
    cmake --build . --target runtest --config %CMAKE_CONFIG%

    if ERRORLEVEL 1 (
        echo ERROR: Tests failed
        exit /b 1
    )

    if %BUILD_PYTHON% EQU 1 (
        if %BUILD_PYTHON_LAYER% EQU 1 (
            :: Run python tests only in Release build since
            :: the _caffe module is _caffe-d is debug
            if "%CMAKE_CONFIG%"=="Release" (
                :: Run the python tests
                cmake --build . --target pytest

                if ERRORLEVEL 1 (
                    echo ERROR: Python tests failed
                    exit /b 1
                )
            )
        )
    )
)

if %RUN_INSTALL% EQU 1 (
    cmake --build . --target install --config %CMAKE_CONFIG%
)

popd
@endlocal