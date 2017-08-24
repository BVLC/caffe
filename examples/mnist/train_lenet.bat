REM going to the caffe root
ECHO OFF
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
SET TOOLS=%ROOT%\build\install\bin
"%TOOLS%\caffe.exe" train --solver=examples\mnist\lenet_solver.prototxt
POPD
pause