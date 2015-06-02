param([bool] $HAVE_CUDNN = 0,
      [bool] $HAVE_CUDA = 0
)

if($HAVE_CUDNN) {
[regex]$rclass = '.*INSTANTIATE_CLASS\((.+)\).*'
[regex]$rlayer = '.*REGISTER_LAYER_CLASS\((.+)\).*'
[regex]$rcreat = '.*REGISTER_LAYER_CREATOR\((.+),.*\).*'
[regex]$rgpu = '.*INSTANTIATE_LAYER_GPU_FUNCS\((.+)\).*'
[regex]$rgpufwd = '.*INSTANTIATE_LAYER_GPU_FORWARD\((.+)\).*'
[regex]$rgpubwd = '.*INSTANTIATE_LAYER_GPU_BACKWARD\(((.+),.*\).*'
}
else {
[regex]$rclass = '.*INSTANTIATE_CLASS\(((?!CuDNN).+)\).*'
[regex]$rlayer = '.*REGISTER_LAYER_CLASS\(((?!CuDNN).+)\).*'
[regex]$rcreat = '.*REGISTER_LAYER_CREATOR\(((?!CuDNN).+),.*\).*'
[regex]$rgpu = '.*INSTANTIATE_LAYER_GPU_FUNCS\(((?!CuDNN).+)\).*'
[regex]$rgpufwd = '.*INSTANTIATE_LAYER_GPU_FORWARD\(((?!CuDNN).+)\).*'
[regex]$rgpubwd = '.*INSTANTIATE_LAYER_GPU_BACKWARD\(((?!CuDNN).+),.*\).*'
}

$sourcefiles = dir *.cpp -Recurse
$classes = $sourcefiles | select-string -Pattern $rclass -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 
$types = $sourcefiles | select-string -Pattern $rlayer -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 
$types += $sourcefiles | select-string -Pattern $rcreat -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 

if($HAVE_CUDA) {
    $cudafiles = dir *.cu -Recurse
    $gpus = $cudafiles | select-string -Pattern $rgpu -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 
    $gpufwds = $cudafiles | select-string -Pattern $rgpufwd -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 
    $gpubwds = $cudafiles | select-string -Pattern $rgpubwd -AllMatches | % { $_.Matches } | % { $_.Groups[1].Value } 
}
else {
    $gpus = @()
    $gpufwds = @()
    $gpubwds = @()
}
@"
#ifndef CAFFE_FORCE_SYMBOL_REFERENCE_H_
#define CAFFE_FORCE_SYMBOL_REFERENCE_H_
namespace caffe {
#ifdef _MSC_VER
"@
foreach($klass in $classes) {
    "  FORCE_INSTANTIATE_CLASS_SYMBOL_REFERENCE($klass);"
}
foreach($type in $types) {
    "  FORCE_REFERENCE_LAYER_CREATOR_SYMBOL_REFERENCE($type);"
}
foreach($gpu in $gpus) {
    "  FORCE_INSTANTIATE_LAYER_GPU_FORWARD_REFERENCE($gpu);"
    "  FORCE_INSTANTIATE_LAYER_GPU_BACKWARD_REFERENCE($gpu);"
}
foreach($gpufwd in $gpufwds) {
    "  FORCE_INSTANTIATE_LAYER_GPU_FORWARD_REFERENCE($gpufwd);"
}
foreach($gpubwd in $gpubwds) {
    "  FORCE_INSTANTIATE_LAYER_GPU_BACKWARD_REFERENCE($gpubwd);"
}
@"
#endif
} // namespace caffe
#endif // CAFFE_FORCE_SYMBOL_REFERENCE_H_
"@