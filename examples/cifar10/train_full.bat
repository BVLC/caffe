REM going to the caffe root
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
SET TOOLS=%ROOT%\build\install\bin
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver.prototxt

REM reduce learning rate by factor of 10
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5

REM reduce learning rate by factor of 10
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
POPD
pause