# Executes scripts to generate code before compiling

set(macro_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

set(layer_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

set(blob_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

set(quantizer_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

set(test_macros_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

set(cuda_nvrtc_header_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")

################################################################################################
function(caffe_prebuild_macros_py output_dir)
    add_custom_command(
        COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/macros.py ${output_dir}
        OUTPUT ${output_dir}/macros.hpp
        COMMENT "Generating macros source (python -> C++)."
    )
endfunction()


function(caffe_prebuild_layer_creator_py output_dir)
	add_custom_command(
		COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/layer_creator.py ${output_dir}
		OUTPUT ${output_dir}/layer_creator.hpp
		COMMENT "Generating layer creator source (python -> C++)."
	)
endfunction()


function(caffe_prebuild_blob_creator_py output_dir)
	add_custom_command(
		COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/blob_creator.py ${output_dir}
		OUTPUT ${output_dir}/blob_creator.hpp
		COMMENT "Generating blob creator source (python -> C++)."
	)
endfunction()


function(caffe_prebuild_quantizer_creator_py output_dir)
	add_custom_command(
		COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/quantizer_creator.py ${output_dir}
		OUTPUT ${output_dir}/quantizer_creator.hpp
		COMMENT "Generating quantizer creator source (python -> C++)."
	)
endfunction()


function(caffe_prebuild_test_macros_py output_dir)
	add_custom_command(
		COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/test_macros.py ${output_dir}
		OUTPUT ${output_dir}/test_macros.hpp
		COMMENT "Generating test macros source (python -> C++)."
	)
endfunction()

function(caffe_cuda_nvrtc_header_loader_py output_dir header_files compiler standard_include_names header_exclude_names)
	add_custom_command(
		COMMAND ${PYTHON_EXECUTABLE} ${Caffe_SCRIPT_DIR}/prebuild/cuda_nvrtc_header_loader.py
			"--output_dir" ${output_dir}
			"--header_files" ${header_files}
			"--compiler" ${compiler}
			"--standard_include_names" ${standard_include_names}
			"--header_exclude_names" ${header_exclude_names}
		OUTPUT ${output_dir}/cuda_nvrtc_headers.hpp
		COMMENT "Loading CUDA NVRTC header files (python -> C++)."
	)
endfunction()