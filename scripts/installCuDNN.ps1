Add-Type -A System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($args[0],'cudnn')
cp -r -force .\cudnn\cuda\* $env:CUDA_PATH
rm -r cudnn
