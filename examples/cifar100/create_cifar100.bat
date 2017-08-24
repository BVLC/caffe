ECHO OFF
REM This script converts the cifar100 data into lmdb format.
REM DeepLearning.ir
SET currentDirectory=%~dp0
PUSHD %CD%
CD ..
CD ..
SET ROOT=%CD%
POPD
SET EXAMPLE=%ROOT%\examples\cifar100
SET DATA=%ROOT%\data\cifar100
SET BUILD=%ROOT%\build\install\bin
SET RM="%ROOT%\tools\3rdparty\bin\rm.exe"
SET DBTYPE=lmdb

ECHO "Creating CIFAR100 %DBTYPE%..."

%rm% -rf %EXAMPLE%\cifar100_train_%DBTYPE% %EXAMPLE%\cifar100_test_%DBTYPE%

"%BUILD%\convert_cifar100_data.exe" %DATA% %EXAMPLE% %DBTYPE%

ECHO "Computing image mean..."

"%BUILD%\compute_image_mean.exe" -backend=%DBTYPE% %EXAMPLE%\cifar100_train_%DBTYPE% %EXAMPLE%\mean.binaryproto

ECHO "Done."
PAUSE