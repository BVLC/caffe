How to refine the inference model

A. Pruning
There are tools to generate pruned model.
Commands list:
	mkdir build_1
	cd build_1
	cmake -DCPU_ONLY=1 -DUSE_MLSL=0 ..
	make all -j
Put the pre-trained model (resnet_50_16_nodes_2k_batch_100_warmup3_iter_56250.caffemodel) in examples/model_optimization/models/Facebook_ResNet_50/Baseline/ directory
	./examples/model_optimization/tools/gen_shapes_Facebook_ResNet_50.sh examples/model_optimization/models/Facebook_ResNet_50/Baseline/train_val.prototxt examples/model_optimization/models/Facebook_ResNet_50/Baseline/resnet_50_16_nodes_2k_batch_100_warmup3_iter_56250.caffemodel Facebook_ResNet_50_train
	python examples/model_optimization/tools/copy_params.py examples/model_optimization/models/Facebook_ResNet_50/Baseline/train_val.prototxt examples/model_optimization/models/Facebook_ResNet_50/Baseline/resnet_50_16_nodes_2k_batch_100_warmup3_iter_56250.caffemodel examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/Facebook_ResNet_50_train.85.prototxt examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/Facebook_ResNet_50_train.85.caffemodel kl
Then we can generate small size Facebook ResNet-50 models in examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/, (i.e. Facebook_ResNet_50_train.85.prototxt, Facebook_ResNet_50_train.85.caffemodel)

B. Fine-tune the pruned model
Command:
	./scripts/run_intelcaffe.sh --hostfile host_2nodes.txt --solver examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/Facebook_ResNet_50_solver.85_2_nodes.prototxt --weights examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/Facebook_ResNet_50_train.85.caffemodel --network opa --benchmark none --engine MKLDNN
Then you can get the fine-tuned pruned model, like Facebook_resnet_50_2_nodes_256_batch_100_warmup3_15p_finest_tune_iter_439600.caffemodel

C. Accuracy Calibration for 8 Bit Inference
Command:
	python calibrator.py -r ../build/ -m examples/model_optimization/models/Facebook_ResNet_50/Facebook_ResNet_50_pruned/Facebook_ResNet_50_train.85.prototxt -w Facebook_resnet_50_2_nodes_256_batch_100_warmup3_15p_finest_tune_iter_439600.caffemodel -i 1000 -l 0.01 -d 0 -c multiple