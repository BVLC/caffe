cmake_minimum_required(VERSION 2.8)
project( Test )
enable_testing()


#    Set CUDA Device number
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_test_device.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_test_device.h
)

include_directories(
        ${gtest_SOURCE_DIR}
        ${gtest_SOURCE_DIR}/include
        ${Caffe_SOURCE_DIR}
)

set(Test_MAIN 
         test_caffe_main.cpp
)

set(Test_SOURCES
        test_benchmark.cpp
        test_blob.cpp       
        test_common.cpp
        test_concat_layer.cpp
        test_convolution_layer.cpp
        test_data_layer.cpp
        test_eltwise_product_layer.cpp
        test_euclidean_loss_layer.cpp
        test_filler.cpp
        test_flatten_layer.cpp
        test_hdf5data_layer.cpp
        test_hdf5_output_layer.cpp
        test_hinge_loss_layer.cpp
        test_im2col_layer.cpp
        test_image_data_layer.cpp
        test_inner_product_layer.cpp
        test_lrn_layer.cpp
        test_math_functions.cpp
        test_memory_data_layer.cpp
        test_multinomial_logistic_loss_layer.cpp
        test_net.cpp
        test_neuron_layer.cpp
        test_platform.cpp
        test_pooling_layer.cpp
        test_power_layer.cpp
        test_protobuf.cpp
        test_random_number_generator.cpp
        test_sigmoid_cross_entropy_loss_layer.cpp
        test_softmax_layer.cpp
        test_softmax_with_loss_layer.cpp
        test_split_layer.cpp
        test_stochastic_pooling.cpp
        test_syncedmem.cpp
        test_tanh_layer.cpp
        test_upgrade_proto.cpp 
        test_util_blas.cpp
)

foreach(source ${Test_SOURCES})

    get_filename_component(name ${source} NAME_WE)
    add_executable(${name}.test ${Test_MAIN} ${source} ../blob.cpp)
    target_link_libraries(${name}.test gtest caffe)
    add_test(${name} ${name}.test)    
    
endforeach()

add_executable(test ${Test_SOURCES} ${Test_MAIN})
target_link_libraries(test gtest caffe)