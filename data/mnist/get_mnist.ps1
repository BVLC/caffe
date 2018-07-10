# This scripts downloads the mnist data and unzips it.
$ErrorActionPreference = 'Stop'

pushd $PSScriptRoot

echo "Downloading..."

# get the path to 7-zip from the registry
$7zip = Join-Path (get-item HKLM:\SOFTWARE\7-Zip).GetValue('Path') '7z.exe'

$fnames = @("train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte")

foreach($fname in $fnames) {
    if(-not (Test-Path $fname)) {
        # Start-BitsTransfer -Source "http://yann.lecun.com/exdb/mnist/$fname.gz" -Destination "$fname.gz"
        wget -Uri "http://yann.lecun.com/exdb/mnist/$fname.gz" -OutFile "$fname.gz"
        . $7zip x "$fname.gz"
    }
}

popd