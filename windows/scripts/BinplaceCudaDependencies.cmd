set CUDA_TOOLKIT_BIN_DIR=%~1%
set CUDNN_PATH=%~2%
set IS_CPU_ONLY_BUILD=%3%
set USE_CUDNN=%4%
set OUTPUT_DIR=%~5%

if %IS_CPU_ONLY_BUILD% == true (
    echo BinplaceCudaDependencies : CPU only build, don't copy cuda dependencies.
 ) else (
    echo BinplaceCudaDependencies : Copy cudart*.dll, cublas*dll, curand*.dll to output.

    copy /y "%CUDA_TOOLKIT_BIN_DIR%\cudart*.dll" "%OUTPUT_DIR%"
    copy /y "%CUDA_TOOLKIT_BIN_DIR%\cublas*.dll" "%OUTPUT_DIR%"
    copy /y "%CUDA_TOOLKIT_BIN_DIR%\curand*.dll" "%OUTPUT_DIR%"

    if %USE_CUDNN% == true (
        echo BinplaceCudaDependencies : Copy cudnn*.dll to output.

        if "%CUDNN_PATH%" == "" (
            copy /y "%CUDA_TOOLKIT_BIN_DIR%\cudnn*.dll" "%OUTPUT_DIR%"
        ) else (
            copy /y "%CUDNN_PATH%\cuda\bin\cudnn*.dll" "%OUTPUT_DIR%"
        )
    ) else (
        echo BinplaceCudaDependencies : cuDNN isn't enabled.
    )
)