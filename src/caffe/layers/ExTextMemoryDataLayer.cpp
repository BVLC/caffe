#include <fstream>
#include <boost/algorithm/string.hpp>

#include "caffe/ExTextMemoryDataLayer.hpp"

using namespace std;
using namespace boost;

namespace caffe {

template <typename Dtype>
void ExTextMemoryDataLayer<Dtype>::addBuffer(const vector<vector<vector<Dtype> > >&
                                      buffer)
{
    vector<size_t> sizes(buffer[0].size());
    for(int row=0;row<buffer.size();row++)
    {
        for (int col=0;col<buffer[row].size();col++)
        {
            sizes[col]=max(sizes[col],buffer[row][col].size());
        }
    }

    vector<Blob<Dtype>* > blobs(buffer[0].size());
    for (int i=0;i<buffer[0].size();i++)
    {
        blobs[i] = new Blob<Dtype>(buffer.size(),sizes[i],1,1);
        for(int j=0;j<blobs[i]->count();j++)
        {
            blobs[i]->mutable_cpu_data()[j]=-1;
        }
    }

    for(int row=0;row<buffer.size();row++)
    {
        for (int col=0;col<buffer[row].size();col++)
        {
            uint64_t start = row*sizes[col];
            memcpy(blobs[col]->mutable_cpu_data()+start,
                   &buffer[row][col][0],
                   buffer[row][col].size()*sizeof(Dtype));
        }
    }

    this->datas_.push_back(blobs);
}

template <typename Dtype>
void ExTextMemoryDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    string filename = this->layer_param_.ex_text_memory_data_param().source();
    int batchSize = this->layer_param_.ex_text_memory_data_param().batch_size();
    
    ifstream infile(filename.c_str());
    CHECK(!infile.fail()) << "Cannot open file: " << filename;

    string line;
    vector<vector<vector<Dtype> > > buffer;
    int lineCount = 0;
    while (!infile.eof())
    {
        line = "";
        getline(infile,line);
        trim(line);

        vector<vector<Dtype> > data;
        if (line.length()>0)
        {
            //cout << lineCount << endl;
            if (lineCount % 100000 == 0)
            {
                LOG(INFO) << "read text data index: " << lineCount;
            }
            lineCount++;
            vector<string> parts;
            split(parts, line, is_any_of(";"));
            if (buffer.size()>0)
            {
                CHECK(buffer[0].size()==parts.size()) << "data size not match";
            }
            for (int i=0;i<parts.size();i++)
            {
                data.push_back(vector<Dtype>());
                vector<Dtype>& subdata = data[data.size()-1];
                vector<string> subparts;
                if (parts[i].length()>0)
                {
                    split(subparts,parts[i],is_any_of(","));
                    for (int j=0;j<subparts.size();j++)
                    {
                        subdata.push_back(std::stof(subparts[j]));
                    }
                }
                else
                {
                    //do nothing
                }
            }
            buffer.push_back(data);
        }//if length>0
        if (buffer.size()==batchSize)
        {
            addBuffer(buffer);
            buffer.clear();
        }
    }//while eof

    infile.close();
    for(int i=0;i<this->layer_param_.top_size();i++)
    {
        top[i]->ReshapeLike(*datas_[0][i]);
    }
}

template<typename Dtype>
void ExTextMemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top)
{
    for (uint32_t i=0;i<top.size();i++)
    {
        Blob<Dtype>* pBlob = this->datas_[this->pos_][i];
        top[i]->ReshapeLike(*pBlob);
        top[i]->ShareData(*pBlob);
    }
    
    this->pos_++;
    if (this->pos_==this->datas_.size())
    {
        this->pos_=0;
    }
}

INSTANTIATE_CLASS(ExTextMemoryDataLayer);
REGISTER_LAYER_CLASS(ExTextMemoryData);

}
