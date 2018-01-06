param(
  [string]$BuildDir
)

$CaffeRoot = (Resolve-Path (Join-Path $PSScriptRoot ..\..))
if("$BuildDir" -eq "") {
  $BuildDir = "$CaffeRoot\build"
}

. $BuildDir\tools\caffe.exe train --solver=examples\mnist\lenet_solver.prototxt $args
