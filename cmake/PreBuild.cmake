# Executes scripts to generate code before compiling

set(macro_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

set(layer_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

set(blob_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

set(quantizer_creator_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

set(test_macros_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

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