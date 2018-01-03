REM going to the caffe root
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
SET TOOLS=%ROOT%\build\install\bin
"%TOOLS%/caffe.exe" train --solver=examples/siamese/mnist_siamese_solver.prototxt
POPD
pause