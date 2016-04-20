set SOLUTION_DIR=%~1%
set OUTPUT_DIR=%~2%

echo MatlabPreBuild.cmd : Create output directories for matlab scripts.

if not exist "%OUTPUT_DIR%\matcaffe" mkdir "%OUTPUT_DIR%\matcaffe"
if not exist "%OUTPUT_DIR%\matcaffe\+caffe" mkdir "%OUTPUT_DIR%\matcaffe\+caffe"
if not exist "%OUTPUT_DIR%\matcaffe\+caffe\private" mkdir "%OUTPUT_DIR%\matcaffe\+caffe\private"
