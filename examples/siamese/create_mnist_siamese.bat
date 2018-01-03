ECHO OFF
ECHO    This script converts the mnist data into lmdb format.

SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
POPD
SET SIAMESE_DIR=%currentDirectory:~0,-1%
SET DATA=%ROOT%\data\mnist
SET BUILD=%ROOT%\build\install\bin
SET BACKEND=lmdb

echo "Creating %BACKEND%..."

if exist "%SIAMESE_DIR%\mnist_siamese_train_%BACKEND%" (
ECHO    "Removing previously created datasets..."

rd /s /q "%SIAMESE_DIR%\mnist_siamese_train_%BACKEND%"
rd /s /q "%SIAMESE_DIR%\mnist_siamese_test_%BACKEND%"
)
ECHO    "Converting..."

"%BUILD%\convert_mnist_siamese_data.exe" %DATA%\train-images-idx3-ubyte %DATA%\train-labels-idx1-ubyte mnist_siamese_train_%BACKEND% 
"%BUILD%\convert_mnist_siamese_data.exe" %DATA%\t10k-images-idx3-ubyte %DATA%\t10k-labels-idx1-ubyte mnist_siamese_test_%BACKEND% 

echo "Done."
PAUSE
