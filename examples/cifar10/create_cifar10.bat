ECHO OFF
ECHO                           Attention
ECHO     lmdb in windows may crash, in case it does!, simlply change
ECHO     lmdb to leveldb. afte the datasets are created, make sure
ECHO     to change the train_test prototxt files as well( change all lmdb to leveldb)
ECHO       
REM This script converts the cifar data into lmdb format.
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
POPD
SET EXAMPLE=%ROOT%\examples\cifar10
SET DATA=%ROOT%\data\cifar10
SET BUILD=%ROOT%\build\install\bin
SET RM="%ROOT%\tools\3rdparty\bin\rm.exe"
SET DBTYPE=lmdb

ECHO "Creating %DBTYPE%..."

%rm% -rf %EXAMPLE%\cifar10_train_%DBTYPE% %EXAMPLE%\cifar10_test_%DBTYPE%

"%BUILD%\convert_cifar_data.exe" %DATA% %EXAMPLE% %DBTYPE%

ECHO "Computing image mean..."

"%BUILD%\compute_image_mean.exe" -backend=%DBTYPE% %EXAMPLE%\cifar10_train_%DBTYPE% %EXAMPLE%\mean.binaryproto

ECHO "Done."
PAUSE