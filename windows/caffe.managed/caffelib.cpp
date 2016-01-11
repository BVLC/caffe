#include "stdafx.h"

using namespace std;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(float));

namespace CaffeLibMC {

  public ref class CaffeModel
  {
  private:
    _CaffeModel *m_net;

  public:
    static int DeviceCount;

    static CaffeModel()
    {
      int count;
      cudaGetDeviceCount(&count);
      DeviceCount = count;
    }

    static void SetDevice(int deviceId)
    {
      _CaffeModel::SetDevice(deviceId);
    }

    CaffeModel(String ^netFile, String ^modelFile)
    {
      m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile));
    }

    // destructor to call finalizer
    ~CaffeModel()
    {
      this->!CaffeModel();
    }

    // finalizer to release unmanaged resource
    !CaffeModel()
    {
      delete m_net;
      m_net = NULL;
    }

    array<float>^ ExtractOutputs(String^ imageFile, int interpolation, String^ blobName)
    {
      FloatArray intermediate = m_net->ExtractOutputs(TO_NATIVE_STRING(imageFile), interpolation, TO_NATIVE_STRING(blobName));
      MARSHAL_ARRAY(intermediate, outputs)
        return outputs;
    }

    array<array<float>^>^ ExtractOutputs(String^ imageFile, int interpolation, array<String^>^ blobNames)
    {
      std::vector<std::string> names;
      for each(String^ name in blobNames)
        names.push_back(TO_NATIVE_STRING(name));
      std::vector<FloatArray> intermediates = m_net->ExtractOutputs(TO_NATIVE_STRING(imageFile), interpolation, names);
      auto outputs = gcnew array<array<float>^>(names.size());
      for (int i = 0; i < names.size(); ++i)
      {
        auto intermediate = intermediates[i];
        MARSHAL_ARRAY(intermediate, values)
          outputs[i] = values;
      }
      return outputs;
    }
  };
}
