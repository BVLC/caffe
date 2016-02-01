:: Glog nuget package has dependency on GFlags nuget package
:: Caffe also has direct dependency on GFlags
:: Unfortunately in GLog nuget package, dependency to GFlags dll was incorrectly set (naming is wrong)
:: For this reasons Caffe needs gflags.dll/gflagsd.dll in release/debug
:: and GLog needs libgflags.dll/libgflags-debug.dll in release/debug
:: This scripts is a workaround for this issue.

set OUTPUT_DIR=%~1%
set BUILD_CONFIG=%2%

if %BUILD_CONFIG% == Release (
    set originalDllName=gflags.dll
    set newDllName=libgflags.dll
) else (
    set originalDllName=gflagsd.dll
    set newDllName=libgflags-debug.dll
)

if exist "%OUTPUT_DIR%\%newDllName%" (
    echo FixGFlagsNaming.cmd : "%newDllName%" already exists
) else (
    echo FixGFlagsNaming.cmd : mklink /H "%OUTPUT_DIR%\%newDllName%" "%OUTPUT_DIR%\%originalDllName%"
    mklink /H "%OUTPUT_DIR%\%newDllName%" "%OUTPUT_DIR%\%originalDllName%"
)