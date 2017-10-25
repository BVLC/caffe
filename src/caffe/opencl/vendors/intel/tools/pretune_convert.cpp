#include <iostream>
#include <fstream>
#include <map>

#define USE_INTEL_SPATIAL
#define USE_GREENTEA
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"

typedef caffe::ConvolutionLayerSpatial<float>::PretunedKey PretunedKey;
typedef caffe::ConvolutionLayerSpatial<float>::PretunedValue PretunedValue;

const char* pre =
"#ifdef USE_INTEL_SPATIAL\n"
"#include \"caffe/layers/conv_spatial_layer.hpp\"\n"
"namespace caffe {\n"
"template<typename Dtype>\n"
"std::map<typename ConvolutionLayerSpatial<Dtype>::PretunedKey,\n"
"         typename ConvolutionLayerSpatial<Dtype>::PretunedValue>\n"
"ConvolutionLayerSpatial<Dtype>::pretuned_kv = {\n";

const char* post =
"};\n"
"#ifdef HAS_HALF_SUPPORT\n"
"template class ConvolutionLayerSpatial<half>;\n"
"#endif\n"
"template class ConvolutionLayerSpatial<float>;\n"
"template class ConvolutionLayerSpatial<double>;\n"
"}\n"
"#endif\n";

static std::map<PretunedKey, PretunedValue> kvs;

void Convert(std::ofstream& fout, const char* infile)
{
  std::cout << "parsing pretuned binary file: " << infile << std::endl;
  std::ifstream fin(infile, std::ios::binary);
  fin.seekg(0, std::ios::end);
  size_t length = fin.tellg();
  if (length == 0) {
    fin.close();
    std::cout << "empty file, ignore it." << std::endl;
    return;
  }

  size_t sizekey = sizeof(PretunedKey);
  size_t sizevalue = sizeof(PretunedValue);
  if (length % (sizekey + sizevalue) != 0) {
    fin.close();
    std::cout << "wrong file size, ignore it." << std::endl;
    return;
  }

  fin.seekg(0, std::ios::beg);
  char* buffer = new char[length];
  fin.read(buffer, length);
  fin.close();

  size_t i = 0;
  size_t count = 0;
  while (i < length) {
    PretunedKey* key = reinterpret_cast<PretunedKey*>(&buffer[i]);
    i += sizekey;
    PretunedValue* value = reinterpret_cast<PretunedValue*>(&buffer[i]);
    i += sizevalue;

    if (kvs.find(*key) != kvs.end()) {
      if (kvs[*key] != *value) {
        std::cout << "there are different values with the same key: " << key->str() << std::endl;
        std::cout << "value in file " << infile << ": " << value->str() << std::endl;
        std::cout << "previouse value: " << kvs[*key].str() << std::endl;
        std::cout << "ignore the later one" << std::endl;
      }
      continue;
    }
    kvs[*key] = *value;

    fout << "{" << key->str() << "," << value->str() << "}," << std::endl;
    ++count;
  }
  std::cout << count << " sets of key-value added." << std::endl;

  delete[] buffer;
}

int main(int argc, char* argv[])
{
  std::ofstream f("conv_layer_spatial_pretuned_data.cpp");
  f << pre;
  for (int i = 1; i < argc; ++i) {
    Convert(f, argv[i]);
  }
  f << post;
  f.close();
  return 0;
}
