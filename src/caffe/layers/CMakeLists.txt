project( Layers )

#	Include directories for easier handling of includes
include_directories( 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${Layers_INCLUDE_DIR}
        ${Proto_INCLUDE_DIR}
 )
 
cuda_add_library(cuda_layers
        bnll_layer.cu
        concat_layer.cu
        conv_layer.cu
        data_layer.cu
        dropout_layer.cu
        eltwise_product_layer.cu
        flatten_layer.cu
        hdf5_data_layer.cu
        hdf5_output_layer.cu
        im2col_layer.cu
        image_data_layer.cu
        inner_product_layer.cu
        lrn_layer.cu
        pooling_layer.cu
        power_layer.cu
        relu_layer.cu
        sigmoid_cross_entropy_loss_layer.cu
        sigmoid_layer.cu
        softmax_layer.cu
        softmax_loss_layer.cu
        split_layer.cu
        tanh_layer.cu
        window_data_layer.cu
)

add_library(layers
        bnll_layer.cpp
        concat_layer.cpp
        conv_layer.cpp
        data_layer.cpp
        dropout_layer.cpp
        eltwise_product_layer.cpp
        flatten_layer.cpp
        hdf5_data_layer.cpp
        hdf5_output_layer.cpp
        im2col_layer.cpp
        image_data_layer.cpp
        inner_product_layer.cpp
        loss_layer.cpp
        lrn_layer.cpp
        memory_data_layer.cpp
        neuron_layer.cpp
        pooling_layer.cpp
        power_layer.cpp
        relu_layer.cpp
        sigmoid_cross_entropy_loss_layer.cpp
        sigmoid_layer.cpp
        softmax_layer.cpp
        softmax_loss_layer.cpp
        split_layer.cpp
        tanh_layer.cpp
        window_data_layer.cpp  
 )
 
 target_link_libraries(layers cuda_layers)
