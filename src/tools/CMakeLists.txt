project( Tools )

set(Tools_SOURCES
    compute_image_mean.cpp
    convert_imageset.cpp
    device_query.cpp
    dump_network.cpp
    extract_features.cpp
    finetune_net.cpp
    net_speed_benchmark.cpp
    test_net.cpp
    train_net.cpp
    upgrade_net_proto_binary.cpp
    upgrade_net_proto_text.cpp
)

foreach(source ${Tools_SOURCES})
    get_filename_component(name ${source} NAME_WE)  
    add_executable(${name} ${source} ../caffe/blob.cpp)
    target_link_libraries(${name} caffe)
endforeach(source)