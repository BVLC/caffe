project( Examples )

set(Examples_SOURCES
        cifar10/convert_cifar_data.cpp
        mnist/convert_mnist_data.cpp
)

foreach(source ${Examples_SOURCES})
    get_filename_component(name ${source} NAME_WE)  
    add_executable(${name} ${source})
    target_link_libraries(${name} caffe)
endforeach(source)