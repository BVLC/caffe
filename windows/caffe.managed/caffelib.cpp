#include "stdafx.h"

using namespace std;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

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
#ifndef CPU_ONLY
      int count;
      cudaGetDeviceCount(&count);
      DeviceCount = count;
#else
      DeviceCount = 0;
#endif
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
      auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
      for (int i = 0; i < names.size(); ++i)
      {
        auto intermediate = intermediates[i];
        MARSHAL_ARRAY(intermediate, values)
          outputs[i] = values;
      }
      return outputs;
    }

    string ConvertToDatum(Bitmap ^imgData)
    {
        string datum_string;

        int width = m_net->GetInputImageWidth();
        int height = m_net->GetInputImageHeight();

        Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

        // resize image
        Bitmap ^resizedBmp;
        if (width == imgData->Width && height == imgData->Height)
        {
            resizedBmp = imgData->Clone(rc, PixelFormat::Format24bppRgb);
        }
        else
        {
            resizedBmp = gcnew Bitmap((Image ^)imgData, width, height);
            resizedBmp = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);
        }
        // get image data block
        BitmapData ^bmpData = resizedBmp->LockBits(rc, ImageLockMode::ReadOnly, resizedBmp->PixelFormat);
        pin_ptr<char> bmpBuffer = (char *)bmpData->Scan0.ToPointer();

        // prepare string buffer to call Caffe model
        datum_string.resize(3 * width * height);
        char *buff = &datum_string[0];
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                int line_offset = h * bmpData->Stride + c;
                for (int w = 0; w < width; ++w)
                {
                    *buff++ = bmpBuffer[line_offset + w * 3];
                }
            }
        }
        resizedBmp->UnlockBits(bmpData);

        return datum_string;
    }

    array<float>^ ExtractOutputs(Bitmap^ imgData, String^ blobName)
    {
        string datum_string = ConvertToDatum(imgData);

        FloatArray intermediate = m_net->ExtractBitmapOutputs(datum_string, 0, TO_NATIVE_STRING(blobName));
        MARSHAL_ARRAY(intermediate, outputs)
            return outputs;
    }

    array<array<float>^>^ ExtractOutputs(Bitmap^ imgData, array<String^>^ blobNames)
    {
        string datum_string = ConvertToDatum(imgData);

        vector<string> names;
        for each(String^ name in blobNames)
            names.push_back(TO_NATIVE_STRING(name));
        vector<FloatArray> intermediates = m_net->ExtractBitmapOutputs(datum_string, 0, names);
        auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
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
