# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
param(
  [string]$BuildDir
)

$ErrorActionPreference = 'Stop'

$CaffeRoot = (Resolve-Path (Join-Path $PSScriptRoot ..\..))
$EXAMPLE = "$CaffeRoot\examples\mnist"
$DATA = "$CaffeRoot\data\mnist"
if("$BuildDir" -eq "") {
  $BuildDir = "$CaffeRoot\build"
}
$BUILD = "$BuildDir\examples\mnist"

$BACKEND = "lmdb"

echo "Creating $BACKEND..."

if(Test-Path $EXAMPLE\mnist_train_$BACKEND) {
  rm -Recurse -Force $EXAMPLE\mnist_train_$BACKEND
}
if(Test-Path $EXAMPLE\mnist_train_$BACKEND) {
  rm -Recurse -Force $EXAMPLE\mnist_test_$BACKEND
}

. $BUILD\convert_mnist_data.exe $DATA\train-images.idx3-ubyte `
  $DATA\train-labels.idx1-ubyte $EXAMPLE\mnist_train_$BACKEND --backend=$BACKEND
. $BUILD\convert_mnist_data.exe $DATA\t10k-images.idx3-ubyte `
  $DATA\t10k-labels.idx1-ubyte $EXAMPLE\mnist_test_$BACKEND --backend=$BACKEND

echo "Done."
