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
"ConvolutionLayerSpatial<Dtype>::pretuned_kv;\n"
"template<typename Dtype>\n"
"std::set<typename ConvolutionLayerSpatial<Dtype>::PretunedValue>\n"
"ConvolutionLayerSpatial<Dtype>::pretuned_vset;\n"
"template<typename Dtype>\n"
"void ConvolutionLayerSpatial<Dtype>::InitPretunedKey(void) {\n";

const char* post =
"}\n"
"#ifdef HAS_HALF_SUPPORT\n"
"template class ConvolutionLayerSpatial<half>;\n"
"#endif\n"
"template class ConvolutionLayerSpatial<float>;\n"
"template class ConvolutionLayerSpatial<double>;\n"
"}\n"
"#endif\n";

struct PretunedValueVersion1 {
  const static int BLOCK_W_BITS = 8;
  const static int BLOCK_H_BITS = 8;
  const static int BLOCK_D_BITS = 8;
  const static int LOCAL_SIZE_X_BITS = 8;
  const static int LOCAL_SIZE_Y_BITS = 8;
  const static int LOCAL_SIZE_Z_BITS = 8;

  const static int KERNEL_TYPE_BITS = 3;

  uint32_t block_w:       BLOCK_W_BITS;
  uint32_t block_h:       BLOCK_H_BITS;
  uint32_t block_d:       BLOCK_D_BITS;
  uint32_t local_size_x:  LOCAL_SIZE_X_BITS;
  uint32_t local_size_y:  LOCAL_SIZE_Y_BITS;
  uint32_t local_size_z:  LOCAL_SIZE_Z_BITS;
  uint32_t kernel_type:   KERNEL_TYPE_BITS;
};

char *convert_version(int count, char *buffer, int fileVersion, int version) {
  if (fileVersion == PretunedVersion) {
    return buffer;
  }
  if (fileVersion != 1 || version != 2) {
    delete [] buffer;
    return NULL;
  }
  if (version == 2 && fileVersion == 1) {

    char *new_buffer;
    size_t fileKeySize = sizeof(PretunedKey);
    size_t fileValueSize = sizeof(PretunedValueVersion1);
    size_t sizekey = sizeof(PretunedKey);
    size_t sizevalue = sizeof(PretunedValue);
    new_buffer = new char[count * (sizekey + sizevalue)];
    size_t i = 0;
    size_t j = 0;
    while(i < (fileKeySize + fileValueSize) * count) {
      auto fileKey = reinterpret_cast<PretunedKey*>(&buffer[i]);
      auto key = reinterpret_cast<PretunedKey*>(&new_buffer[j]);
      *key = *fileKey;
      i += fileKeySize;
      j += sizekey;
      auto fileValue = reinterpret_cast<PretunedValueVersion1*>(&buffer[i]);
      auto value = reinterpret_cast<PretunedValue*>(&new_buffer[j]);
      value->block_w = fileValue->block_w;
      value->block_h = fileValue->block_h;
      value->block_d = fileValue->block_d;
      value->kernel_type = fileValue->kernel_type;
      i += fileValueSize;
      j += sizevalue;
    }
    return new_buffer;
  }
  return NULL;
}

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

  caffe::PretunedMagicNumType magic;
  caffe::PretunedVersionType fileVersion;
  
  fin.seekg(0, std::ios::beg);

  fin.read(static_cast<char*>(static_cast<void*>(&magic)), sizeof(magic));
  fin.read(static_cast<char*>(static_cast<void*>(&fileVersion)), sizeof(fileVersion));
  if (magic != PretunedMagicNum) {
    fin.close();
    std::cout << "wrong magic number ." << std::endl;
    return;
  }

  size_t fileKeySize;
  size_t fileValueSize;

  if (fileVersion != PretunedVersion) {
    if (fileVersion == 1 && PretunedVersion == 2) {
      fileKeySize = sizeof(PretunedKey);
      fileValueSize = sizeof(PretunedValueVersion1);
    } else {
      fin.close();
      std::cout << "incompatible version." << std::endl;
      return;
    }
  } else {
    fileKeySize = sizekey;
    fileValueSize = sizevalue;
  }

  length -= sizeof(caffe::PretunedVersionType) +
            sizeof(caffe::PretunedMagicNumType);
  if ((length % (fileKeySize + fileValueSize)) != 0) {
    fin.close();
    std::cout << "wrong file size, ignore it." << std::endl;
    return;
  }

  char* buffer = new char[length];
  fin.read(buffer, length);
  fin.close();
  if (fileVersion != PretunedVersion) {
    buffer = convert_version(length / (fileKeySize + fileValueSize),
                             buffer, fileVersion, PretunedVersion);
    length = (length / (fileKeySize + fileValueSize)) * (sizekey + sizevalue);
    if (buffer == NULL) {
      std::cout << "Failed to convert data version from " << fileVersion
                << " to " << PretunedVersion << std::endl
                << "Skipping..." << std::endl;
      return;
    }
  }

  size_t i = 0;
  size_t count = 0;
  std::set<PretunedValue> value_sets[2];
  std::string types[2] = {"half", "float"};
  std::map<PretunedKey, PretunedValue> maps[2];
  maps[0].clear();
  maps[1].clear();
  value_sets[0].clear();
  value_sets[1].clear();
  while (i < length) {
    //PretunedKey*
    auto key = reinterpret_cast<PretunedKey*>(&buffer[i]);
    i += sizekey;
    auto value = reinterpret_cast<PretunedValue*>(&buffer[i]);
    i += sizevalue;
    int map_idx = key->data_type;
    if (map_idx != 0 && map_idx != 1) {
      std::cout << "Damaged record with wrong data type. Skipping..." << std::endl;
      delete[] buffer;
      return;
    }
    if (maps[map_idx].find(*key) != maps[map_idx].end()) {
      if (maps[map_idx][*key] != *value) {
        std::cout << "there are different values with the same key: " << key->str() << std::endl;
        std::cout << "value in file " << infile << ": " << value->str() << std::endl;
        std::cout << "previouse value: " << maps[map_idx][*key].str() << std::endl;
        std::cout << "ignore the later one" << std::endl;
      }
      continue;
    }
    maps[map_idx].insert(make_pair(*key, *value));
    value_sets[map_idx].insert(*value);

    ++count;
  }

  for(int j = 0; j < 2; j++) {
    fout << "if (std::is_same<Dtype, " << types[j] << ">::value) {" << std::endl;
    for (auto& kv : maps[j])
      fout << "  pretuned_kv.insert(std::pair<PretunedKey, PretunedValue>(" << kv.first.str() << "," << kv.second.str() << "));" << std::endl;
    for (auto& v : value_sets[j])
      fout << "  pretuned_vset.insert( "<< v.str() << " );" << std::endl;
    fout << "}" << std::endl;
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
