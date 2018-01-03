ECHO OFF
REM This script converts the mnist data into leveldb format,
SET currentDirectory=%~dp0
SET ROOT=%currentDirectory:~0,-15%
SET MNIST_DIR=%currentDirectory:~0,-1%
SET DATA=%ROOT%data\mnist
SET BUILD=%ROOT%Build\x64\Release
SET BACKEND=leveldb


echo "Creating %BACKEND%..."

rd /s /q "%MNIST_DIR%\mnist_train_%BACKEND%"
rd /s /q "%MNIST_DIR%\mnist_test_%BACKEND%"

"%BUILD%\convert_mnist_data.exe" %DATA%\train-images-idx3-ubyte %DATA%\train-labels-idx1-ubyte mnist_train_%BACKEND% --backend=%BACKEND%
"%BUILD%\convert_mnist_data.exe" %DATA%\t10k-images-idx3-ubyte %DATA%\t10k-labels-idx1-ubyte mnist_test_%BACKEND% --backend=%BACKEND%

echo "Done."
PAUSE
