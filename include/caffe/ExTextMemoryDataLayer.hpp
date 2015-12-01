#ifndef CAFFE_EXTEXTMEMORYDATA_HPP_
#define CAFFE_EXTEXTMEMORYDATA_HPP_

#include "data_layers.hpp"

namespace caffe
{
template <typename Dtype>
class ExTextMemoryDataLayer : public BaseDataLayer<Dtype>
{
public:
    explicit ExTextMemoryDataLayer(const LayerParameter& param)
        : BaseDataLayer<Dtype>(param),pos_(0){};

    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "ExTextMemoryData"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

private:
    void addBuffer(const vector<vector<vector<Dtype> > >& buffer);

private:
    vector<vector<Blob<Dtype>* > > datas_;
    uint64_t pos_;
};

}//namespace caffe

#endif
