@setlocal EnableDelayedExpansion
@echo on

@set caffe_dir=%~dp0
@set anaconda_dir=C:\ProgramData\Anaconda2
@if exist %anaconda_dir%\Scripts\activate.bat (
    call %anaconda_dir%\Scripts\activate.bat %anaconda_dir%
    @set python_version=2 
) else (
    set anaconda_dir=C:\ProgramData\Anaconda3
    if exist !anaconda_dir!\Scripts\activate.bat (
        call !anaconda_dir!\Scripts\activate.bat !anaconda_dir!
        @set python_version=3
    ) else (
        set anaconda_dir="C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda2_64"
        if exist !anaconda_dir!\Scripts\activate.bat (
            call !anaconda_dir!\Scripts\activate.bat !anaconda_dir!
            @set python_version=2
        ) else (
            set anaconda_dir="C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64"
            if exist !anaconda_dir!\Scripts\activate.bat (
                call !anaconda_dir!\Scripts\activate.bat !anaconda_dir!
                @set python_version=3
            ) else (
                goto error2
            )
        )
    )
)

for /R "C:\Program Files (x86)\Microsoft Visual Studio\2017" %%i in (vcvarsall.bat) do @if exist %%i @call "%%i" x64
if %errorlevel% neq 0 goto error

set boost_installed=Y
set /p boost_installed=Have Installed Boost? [Y/n]:
set boost_dir=c:\local\boost_1_69_0
set /p boost_dir=Boost Path is at ? [c:\local\boost_1_69_0]:

if not exist %boost_dir% (
    mkdir %boost_dir%
)

set boost_url=https://ayera.dl.sourceforge.net/project/boost/boost-binaries/1.69.0/boost_1_69_0-msvc-14.1-64.exe
set boost_prebuilt_bin_path=c:\Temp\boost_1_69_0-msvc-14.1-64.exe
if %boost_installed%==n (
    if not exist %boost_prebuilt_bin_path% (
      @echo Boost prebuilt binary installation file is not found at %boost_prebuilt_bin_path%, Downloading it now.
      bitsadmin /transfer DownloadBoost /download /priority normal %boost_url% %boost_prebuilt_bin_path%
      if not exist %boost_prebuilt_bin_path% (
        @echo Boost download fails. Please manually download it from %boost_url% and save it to c:\Temp directory.
        goto exit
      )
    )
    msiexec /i %boost_prebuilt_bin_path% /quiet
)

@echo Ensure you have boost.python dll version. Otherwise please rebuild it by enter Y at below!
set boost_python_rebuilt=N
set /p boost_python_rebuilt=Rebuild Boost Python with current python env? [y/N]:
if %boost_python_rebuilt%==y (
    @echo Building boost.python static lib for pycaffe
    pushd %boost_dir% 
    call .\bootstrap.bat
    pushd %boost_dir%
    rem Building boost.python as dynamic library is mandotory to avoid the error of "TypeError: No to_python (by-value) converter found for C++ type: class caffe::LayerParameter" when running pycaffe
    rem For details, pls refer to https://github.com/BVLC/caffe/issues/3915
    .\bjam.exe --with-python threading=multi link=shared address-model=64 variant=release
    copy .\stage\lib\* .\lib64-msvc-14.1\
    popd
    popd
)

set cmake_exe_url=https://github.com/Kitware/CMake/releases/download/v3.14.1/cmake-3.14.1-win64-x64.msi
set cmake_install_exe_path=c:\Temp\cmake-3.14.1-win64-x64.msi
"C:\Program Files\CMake\bin\cmake.exe" --version
if %errorlevel% neq 0 (
    if not exist %cmake_install_exe_path% (
      @echo cmake installation file is not found at %cmake_install_exe_path%. Downloading it now.
      bitsadmin /transfer DownloadCmake /download /priority normal %cmake_exe_url% %cmake_install_exe_path%
      if not exist %cmake_install_exe_path% (
        @echo cmake download fails. Please manually download it from %cmake_exe_url% and save it to c:\Temp directory.
        goto exit
      )
    )
    msiexec /i c:\Temp\cmake-3.14.1-win64-x64.msi /quiet
)
set "path=C:\Program Files\CMake\bin\;%path%"

set third_party=%cd%\third_party

set gflags_dir=%third_party%\gflags
if not exist %gflags_dir% (
    git clone https://github.com/gflags/gflags.git %gflags_dir%
    pushd %gflags_dir%
    cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -DGFLAGS_BUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set glog_dir=%third_party%\glog
if not exist %glog_dir% (
    git clone https://github.com/google/glog %glog_dir%
    pushd %glog_dir%
    cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set protobuf_dir=%third_party%\protobuf
if not exist %protobuf_dir% (
    git clone  https://github.com/google/protobuf.git --branch=v3.5.2 %protobuf_dir%
    pushd %protobuf_dir%
    git submodule update --init --recursive
    rem Building protobuf as dynamic library would bring issue when running caffe time with a topology containing python layer.
    rem The issue is something like "[libprotobuf ERROR google/protobuf/descriptor_database.cc:58] File already exists in database: caffe.proto".
    rem For details, pls refer to https://github.com/BVLC/caffe/issues/1917
    cd cmake && cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set PATH=%PATH%;%third_party%\bin

set hdf5_dir=%third_party%\hdf5
if not exist %hdf5_dir% (
    git clone https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git --branch=hdf5-1_10_4 --depth=1 %hdf5_dir%
    pushd %hdf5_dir%
    mkdir build
    cd build && cmake -G "Visual Studio 15 2017 Win64" --config Release --target install .. -DCMAKE_INSTALL_PREFIX=%third_party% -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set lmdb_dir=%third_party%\lmdb
if not exist %lmdb_dir% (
    git clone https://github.com/LMDB/lmdb.git %lmdb_dir%
    pushd %lmdb_dir%\libraries\liblmdb
    copy %third_party%\lmdb_CMakeLists.txt CMakeLists.txt
    cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set leveldb_dir=%third_party%\leveldb
if not exist %leveldb_dir% (
    git clone https://github.com/google/leveldb.git --branch=v1.20 %leveldb_dir%
    pushd %leveldb_dir%
    copy %third_party%\leveldb_win.patch .
    git apply leveldb_win.patch
    cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set snappy_dir=%third_party%\snappy
if not exist %snappy_dir% (
    git clone https://github.com/google/snappy.git %snappy_dir%
    pushd %snappy_dir%
    cmake -G "Visual Studio 15 2017 Win64" --build build --config Release --target install . -DCMAKE_INSTALL_PREFIX=%third_party% -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

set opencv_dir=%third_party%\opencv-3.4.5
if not exist %opencv_dir% (
    git clone https://github.com/opencv/opencv.git --branch=3.4.5 %opencv_dir%
    pushd %opencv_dir%
    mkdir build
    cd build && cmake -G "Visual Studio 15 2017 Win64" --build . --config Release --target install .. -DCMAKE_INSTALL_PREFIX=%third_party% -DWITH_FFMPEG=OFF -D WITH_IPP=OFF -DBUILD_SHARED_LIBS=1
    msbuild /p:Configuration=Release install.vcxproj
    popd
)

if not exist build (
    mkdir build
)
cd build && cmake -G "Visual Studio 15 2017 Win64" --config Release .. -DCMAKE_INCLUDE_PATH=%third_party%\include -DCMAKE_LIBRARY_PATH=%third_party%\bin;%third_party%\lib -DUSE_MLSL=OFF -Dpython_version=%python_version%
msbuild /p:Configuration=Release install.vcxproj
cd ..

@echo Don't forget to setting PATH ^& PYTHONPATH envirnoment variables and activating anaconda env like below before run!!!
@echo 1^) set PATH=%cd%\build\install\lib;%boost_dir%\lib64-msvc-14.1;%third_party%\bin;%third_party%\x64\vc15\bin;%cd%\external\mkl\mklml_win_2019.0.3.20190220\lib;%cd%\external\mkldnn\install\bin;%%PATH%%
@echo 2^) set PYTHONPATH=%cd%\build\install\python
@echo 3^) call %anaconda_dir%\Scripts\activate.bat

goto exit

:error
@echo "vcvarsall.bat not found! Please ensure you have installed VS2017."
goto exit

:error2
@echo "activate.bat of anaconda2/3 not found in their default locations! Please ensure you have installed anaconda."

:exit
@endlocal
