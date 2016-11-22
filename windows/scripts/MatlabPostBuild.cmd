set SOLUTION_DIR=%~1%
set OUTPUT_DIR=%~2%

echo MatlabPostBuild.cmd : copy matlab generated scripts to output.

@echo run_tests.m > "%temp%\excludelist.txt"
xcopy /y "%SOLUTION_DIR%..\matlab\+caffe\*.m" "%OUTPUT_DIR%matcaffe\+caffe" /exclude:%temp%\excludelist.txt
copy /y "%SOLUTION_DIR%..\matlab\+caffe\private\*.m" "%OUTPUT_DIR%matcaffe\+caffe\private"
move /y "%OUTPUT_DIR%caffe_.*" "%OUTPUT_DIR%matcaffe\+caffe\private"
