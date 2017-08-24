ECHO OFF
REM This script converts the cifar data into leveldb format.
SET currentDirectory=%~dp0
SET ROOT=%currentDirectory:~0,-18%
SET EXAMPLE=%ROOT%\examples\cifar10
SET DATA=%ROOT%\data\cifar10
SET BUILD=%ROOT%\Build\x64\Release
SET RM="%ROOT%\tools\3rdparty\bin\rm.exe"
SET DBTYPE=leveldb

ECHO "Creating %DBTYPE%..."

%rm% -rf %EXAMPLE%\cifar10_train_%DBTYPE% %EXAMPLE%\cifar10_test_%DBTYPE%

"%BUILD%\convert_cifar_data.exe" %DATA% %EXAMPLE% %DBTYPE%

ECHO "Computing image mean..."

"%BUILD%\compute_image_mean.exe" -backend=%DBTYPE% %EXAMPLE%\cifar10_train_%DBTYPE% %EXAMPLE%\mean.binaryproto

ECHO "Done."
PAUSE