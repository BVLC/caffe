# Executes scripts to generate code before compiling

set(macro_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/")
include_directories("${PROJECT_BINARY_DIR}/include")

################################################################################################
function(caffe_prebuild_macros_py output_dir)

    add_custom_command(
        COMMAND ${PYTHON_EXECUTABLE} ${Caffe_INCLUDE_DIR}/caffe/macros.py ${output_dir}
        OUTPUT ${output_dir}/macros.hpp
        COMMENT "Generating macros."
    )
    
endfunction()
