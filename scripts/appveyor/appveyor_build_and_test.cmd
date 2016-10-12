if "%WITH_CMAKE%" == "1" (
    echo "Building with CMake"
    call %~dp0appveyor_cmake_build_and_test.cmd
) else (
    echo "Building with Visual Studio"
    call %~dp0appveyor_vs_build_and_test.cmd
)
