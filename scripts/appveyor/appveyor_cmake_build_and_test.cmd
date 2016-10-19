@echo off

:: Set python 2.7 with conda as the default python
set PATH=C:\Miniconda-x64;C:\Miniconda-x64\Scripts;C:\Miniconda-x64\Library\bin;%PATH%
:: Check that we have the right python version
python --version
:: Add the required channels
conda config --add channels conda-forge
conda config --add channels willyd
:: Update conda
conda update conda -y
:: Create an environment
:: Todo create protobuf package for vc14
conda install --yes cmake ninja numpy scipy protobuf==3.1.0.vc12 six scikit-image

:: Create build directory and configure cmake
mkdir build
pushd build
:: Download dependencies from VS x64
python ..\scripts\download_prebuilt_dependencies.py --msvc_version v%MSVC_VERSION%0
:: Add the dependencies to the PATH
:: Prepending is crucial since the hdf5 dll may conflict with python's
call %cd%\libraries\prependpath.bat
:: Setup the environement for VS x64
@setlocal EnableDelayedExpansion
set batch_file=!VS%MSVC_VERSION%0COMNTOOLS!..\..\VC\vcvarsall.bat
@endlocal & set batch_file=%batch_file%
call "%batch_file%" amd64
:: Configure using cmake and using the caffe-builder dependencies
cmake -G"%CMAKE_GENERATOR%" ^
      -DBLAS=Open ^
      -DCMAKE_BUILD_TYPE=%CMAKE_CONFIG% ^
      -DBUILD_SHARED_LIBS=%CMAKE_BUILD_SHARED_LIBS% ^
      -C libraries\caffe-builder-config.cmake ^
      ..\

:: Build the library and tools
cmake --build . --config %CMAKE_CONFIG%

if ERRORLEVEL 1 (
  echo Build failed
  exit /b 1
)

:: Build and exectute the tests
if "%CMAKE_BUILD_SHARED_LIBS%"=="OFF" (
  :: Run the tests only for static lib as the shared lib is causing an issue.
  cmake --build . --target runtest --config %CMAKE_CONFIG%

  if ERRORLEVEL 1 (
    echo Tests failed
    exit /b 1
  )

  :: Run python tests only in Release build since
  :: the _caffe module is _caffe-d is debug
  if "%CMAKE_CONFIG%"=="Release" (
    :: Run the python tests
    cmake --build . --target pytest

    if ERRORLEVEL 1 (
      echo Python tests failed
      exit /b 1
    )
  )
)

:: Lint
cmake --build . --target lint  --config %CMAKE_CONFIG%

if ERRORLEVEL 1 (
  echo Lint failed
  exit /b 1
)

popd