// SWIG interface
 %module jaffe
 
 %{
 #include "caffe/proto/caffe.pb.h"
 #include "caffe/proto/caffe_pretty_print.pb.h"
 #include "caffe/blob.hpp"
 #include "caffe/caffe.hpp"
 #include "caffe/common.hpp"
 #include "caffe/data_layers.hpp"
 #include "caffe/filler.hpp"
 #include "caffe/layer.hpp"
 #include "caffe/loss_layers.hpp"
 #include "caffe/net.hpp"
 #include "caffe/neuron_layers.hpp"
 #include "caffe/solver.hpp"
 #include "caffe/syncedmem.hpp"
 #include "caffe/vision_layers.hpp"
 #include "caffe/util/benchmark.hpp"
 #include "caffe/util/im2col.hpp"
 #include "caffe/util/insert_splits.hpp"
 #include "caffe/util/io.hpp"
 #include "caffe/util/math_functions.hpp"
 #include "caffe/util/mkl_alternate.hpp"
 #include "caffe/util/rng.hpp"
 #include "caffe/util/upgrade_proto.hpp"
 %}
 

%include <std_map.i>
%include <std_pair.i> 
%include <std_string.i>
%include <std_vector.i> 
%include <boost_shared_ptr.i>


%define LIBPROTOBUF_EXPORT
%enddef 
%import "/usr/include/google/protobuf/message_lite.h"
%import "/usr/include/google/protobuf/message.h"

%rename(CopyFrom) operator=;
%rename(caffeReadProtoFromTextFileStringMessage) caffe::ReadProtoFromTextFile(string const &,Message *);
%rename(caffeReadProtoFromTextFileOrDieStringMessage) caffe::ReadProtoFromTextFileOrDie(string const &,Message *);
%rename(caffeWriteProtoToTextFileMessageString) caffe::WriteProtoToTextFile(Message const &,string const &);
%rename(caffeReadProtoFromBinaryFileStringMessage) caffe::ReadProtoFromBinaryFile(string const &,Message *);
%rename(caffeReadProtoFromBinaryFileOrDieStringMessage) caffe::ReadProtoFromBinaryFileOrDie(string const &,Message *);
%rename(caffeWriteProtoToBinaryFileMessageString) caffe::WriteProtoToBinaryFile(Message const &,string const &);
%rename(caffeNetParameterPrettyPrintSetNameCharConst) caffe::NetParameterPrettyPrint::set_name(char const *);
%rename(caffeNetParameterPrettyPrintSetInputIntCharConst) caffe::NetParameterPrettyPrint::set_input(int,char const *);
%rename(caffeNetParameterPrettyPrintAddInputStringConst) caffe::NetParameterPrettyPrint::add_input(::std::string const &);
%rename(caffeprotobuf_AddDesc_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto) caffe::protobuf_AddDesc_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto;
%rename(caffeprotobuf_AssignDesc_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto) caffe::protobuf_AssignDesc_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto;
%rename(caffeprotobuf_ShutdownFile_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto) caffe::protobuf_ShutdownFile_caffe_2fproto_2fcaffe_5fpretty_5fprint_2eproto;
%rename(caffeprotobuf_AddDesc_caffe_2fproto_2fcaffe_2eproto) caffe::protobuf_AddDesc_caffe_2fproto_2fcaffe_2eproto;
%rename(caffeprotobuf_AssignDesc_caffe_2fproto_2fcaffe_2eproto) caffe::protobuf_AssignDesc_caffe_2fproto_2fcaffe_2eproto;
%rename(caffeprotobuf_ShutdownFile_caffe_2fproto_2fcaffe_2eproto) caffe::protobuf_ShutdownFile_caffe_2fproto_2fcaffe_2eproto;
%rename(caffeDatumSetDataCharConst) caffe::Datum::set_data(char const *);
%rename(caffeFillerParameterSetTypeCharConst) caffe::FillerParameter::set_type(char const *);
%rename(caffeNetParameterSetnameCharConst) caffe::NetParameter::set_name(char const *);
%rename(caffeNetParameterSetInputIntCharConst) caffe::NetParameter::set_input(int,char const *);
%rename(caffeNetParameterAddInputCharConst) caffe::NetParameter::add_input(char const *);
%ignore caffe::SolverParameter::set_train_net(char const *);
%ignore caffe::SolverParameter::set_test_net(int,char const *);
%ignore caffe::SolverParameter::add_test_net(char const *);
%ignore caffe::SolverParameter::set_lr_policy(char const *);
%ignore caffe::SolverParameter::set_snapshot_prefix(char const *);
%ignore caffe::SolverState::set_learned_net(char const *);
%ignore caffe::LayerParameter::set_bottom(int,char const *);
%ignore caffe::LayerParameter::add_bottom(char const *);
%ignore caffe::LayerParameter::set_top(int,char const *);
%ignore caffe::LayerParameter::add_top(char const *);
%ignore caffe::LayerParameter::set_name(char const *);
%ignore caffe::DataParameter::set_source(char const *);
%ignore caffe::DataParameter::set_mean_file(char const *);
%ignore caffe::WindowDataParameter::set_source(char const *);
%ignore caffe::WindowDataParameter::set_mean_file(char const *);
%ignore caffe::WindowDataParameter::set_crop_mode(char const *);
%ignore caffe::V0LayerParameter::set_name(char const *);
%ignore caffe::V0LayerParameter::set_type(char const *);
%ignore caffe::V0LayerParameter::set_source(char const *);
%ignore caffe::V0LayerParameter::set_meanfile(char const *);
%ignore caffe::V0LayerParameter::set_det_crop_mode(char const *);
%ignore caffe::HDF5DataParameter::set_source(char const *);
%ignore caffe::HDF5OutputParameter::set_file_name(char const *);
%ignore caffe::ImageDataParameter::set_source(char const *);
%ignore caffe::ImageDataParameter::set_mean_file(char const *);
%ignore caffe::InfogainLossParameter::set_source(char const *);

%ignore PoolingParameter::PoolMethod_Name;

// Parse the original header files
%import "caffe/proto/caffe.pb.h"
%import "caffe/proto/caffe_pretty_print.pb.h"
%include "caffe/blob.hpp"
%include "caffe/caffe.hpp"
%include "caffe/common.hpp"
%include "caffe/data_layers.hpp"
%include "caffe/filler.hpp"
%include "caffe/layer.hpp"
%include "caffe/loss_layers.hpp"
%include "caffe/net.hpp"
%include "caffe/neuron_layers.hpp"
%include "caffe/solver.hpp"
%include "caffe/syncedmem.hpp"
%include "caffe/vision_layers.hpp"
%include "caffe/util/benchmark.hpp"
%include "caffe/util/im2col.hpp"
%include "caffe/util/insert_splits.hpp"
%include "caffe/util/io.hpp"
%include "caffe/util/math_functions.hpp"
%include "caffe/util/mkl_alternate.hpp"
%include "caffe/util/rng.hpp"
%include "caffe/util/upgrade_proto.hpp"
 