REM go to the caffe root
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
SET TOOLS=%ROOT%\build\install\bin
"%TOOLS%/caffe.exe" train --solver=examples/cifar100/cifar100_quick_solver.prototxt
POPD
pause
