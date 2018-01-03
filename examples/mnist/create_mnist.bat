ECHO OFF
ECHO                       ATTENTION
ECHO    This script converts the mnist data into lmdb/leveldb format,
ECHO    depending on the value assigned to BACKEND.
ECHO    lmdb sometimes crashes on windows, if this happens, use leveldb
ECHO    and make sure to change your prototxt to address this change (i.e change all lmdbs to leveldb!)
ECHO      

SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
POPD
SET MNIST_DIR=%currentDirectory:~0,-1%
SET DATA=%ROOT%\data\mnist
SET BUILD=%ROOT%\build\install\bin
SET BACKEND=lmdb

echo %ROOT%
echo %MNIST_DIR%
echo "Creating %BACKEND%..."

rd /s /q "%MNIST_DIR%\mnist_train_%BACKEND%"
rd /s /q "%MNIST_DIR%\mnist_test_%BACKEND%"

"%BUILD%\convert_mnist_data.exe" %DATA%\train-images-idx3-ubyte %DATA%\train-labels-idx1-ubyte mnist_train_%BACKEND% --backend=%BACKEND%
"%BUILD%\convert_mnist_data.exe" %DATA%\t10k-images-idx3-ubyte %DATA%\t10k-labels-idx1-ubyte mnist_test_%BACKEND% --backend=%BACKEND%

echo "Done."
PAUSE
