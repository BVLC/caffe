cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair

cdef extern from "boost/shared_ptr.hpp" namespace "boost":
    cdef cppclass shared_ptr[T]:
        T* get()
        void reset(T*)

cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass RuntimeParameter:
        NumpyDataParameter* mutable_numpy_data_param()
        NumpyDataParameter numpy_data_param()
        bool SerializeToString(string*)
        string DebugString()
    cdef cppclass NumpyDataParameter:
        void add_data(float data)
        void add_shape(unsigned int shape)
        string DebugString()
    cdef cppclass LayerParameter:
        bool ParseFromString(string& data)
        bool SerializeToString(string*)
        RuntimeParameter* mutable_rp()
        string name()
        string& bottom(int)
        int bottom_size()
    enum Phase:
        TRAIN = 0
        TEST = 1

cdef extern from "caffe/tensor.hpp" namespace "caffe":
    cdef cppclass Tensor[float]:
        Tensor()
        Tensor(vector[int]&)
        vector[int] shape()
        int count()
        float* mutable_cpu_mem() except +
        float* mutable_gpu_mem() except +
        void Reshape(vector[int]& shape) except +
        void AddFrom(Tensor& other) except +
        void AddFromGPUPointer(float* px, long long size) except +
        void MulFrom(Tensor& other) except +
        void AddMulFrom(Tensor& other, float alpha) except +
        float DotPFrom(Tensor& other) except +
        void SetValues(float value) except +
        void scale(float value) except +
        void CopyFrom(Tensor& other) except +
        void CopyChunkFrom(Tensor& other, int count, int this_offset, int other_offset) except +

cdef extern from "caffe/blob.hpp" namespace "caffe":
    cdef cppclass Blob[float]:
        Blob()
        Blob(vector[int]&)
        vector[int] shape()
        int count()
        void Reshape(vector[int]& shape) except +
        float* mutable_cpu_data() except +
        float* mutable_cpu_diff() except +
        void ShareData(Blob& other)
        void ShareDiff(Blob& other)
        shared_ptr[Tensor] data()
        shared_ptr[Tensor] diff()

cdef extern from "caffe/layer.hpp" namespace "caffe":
    cdef cppclass Layer[float]:
        float Forward(vector[Blob*]& bottom, vector[Blob*]& top)
        float Backward(vector[Blob*]& top, vector[bool]& propagate_down, vector[Blob*]& bottom)
        LayerParameter& layer_param()
        vector[shared_ptr[Blob]]& blobs()
        vector[shared_ptr[Blob]]& buffers()

cdef extern from "caffe/apollonet.hpp" namespace "caffe":
    cdef cppclass ApolloNet[float]:
        ApolloNet()
        float ForwardLayer(string layer_param_string) except +
        void BackwardLayer(string layer_name) except +
        void ResetForward()
        float DiffL2Norm()
        map[string, shared_ptr[Blob]]& blobs()
        map[string, shared_ptr[Layer]]& layers()
        map[string, shared_ptr[Blob]]& params()
        map[string, float]& param_decay_mults()
        map[string, float]& param_lr_mults()
        void set_phase_test()
        void set_phase_train()
        Phase phase()
        void CopyTrainedLayersFrom(string trained_filename) except +
        void SaveTrainedLayersTo(string trained_filename) except + 
        vector[string]& active_layer_names()
        set[string]& active_param_names()

cdef extern from "caffe/layer_factory.hpp" namespace "caffe::LayerRegistry<float>":
    cdef shared_ptr[Layer] CreateLayer(LayerParameter& param)

