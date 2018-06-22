mkdir examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned
cp $1 examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/$3.100.prototxt
cp $2 examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/$3.100.caffemodel
echo generate small size Facebook ResNet-50 models in examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/
for ((i=85; i<86; ++i))
do
    echo Generating 0.$i model
    cat $1 |awk -v prec=0.$i -f examples/model_optimization/tools/prune_shape.awk > examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/$3.$i.prototxt
    cp examples/model_optimization/models/Facebook_ResNet_50/Baseline/solver_Facebook_ResNet_50_dummy.prototxt examples/model_optimization/tools/Facebook_ResNet_50_dummy.prototxt
    echo net: \"examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/$3.$i.prototxt\" >> examples/model_optimization/tools/Facebook_ResNet_50_dummy.prototxt
    echo snapshot_prefix: \"examples/model_optimization/tools/Facebook_ResNet_50_dummy\" >> examples/model_optimization/tools/Facebook_ResNet_50_dummy.prototxt
    ./build_1/tools/caffe train --solver=examples/model_optimization/tools/Facebook_ResNet_50_dummy.prototxt --engine=MKLDNN >/dev/null 2>/dev/null
    mv examples/model_optimization/tools/Facebook_ResNet_50_dummy_iter_1.caffemodel examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/$3.$i.caffemodel
    rm examples/model_optimization/tools/Facebook_ResNet_50_dummy_iter_1.solverstate
    rm examples/model_optimization/tools/Facebook_ResNet_50_dummy.prototxt
done
