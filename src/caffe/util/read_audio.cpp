#ifdef USE_AUDIO
#include <sndfile.h>

#include <string>
#include <valarray>

#include "caffe/common.hpp"
#include "caffe/util/read_audio.hpp"


namespace caffe {

    int ReadAudioFile(const std::string& filePath, float* data, int capacity,
                      int offset) {
        SF_INFO info = SF_INFO();

        SNDFILE* file = sf_open(filePath.c_str(), SFM_READ, &info);
        CHECK_EQ(sf_error(file), SF_ERR_NO_ERROR) << "Can't open file '"
          << filePath << "': " << sf_strerror(file);

        sf_count_t status = sf_seek(file, offset, SEEK_SET);
        CHECK_NE(status, -1) << "Can't seek to offset in: '" << filePath <<
          "': " << sf_strerror(file);

        sf_count_t numberOfFrames;
        if (info.channels != 1) {
          //  Non-mono audio files will only have first channel read
          std::valarray<float> tempData(info.channels * capacity);
          numberOfFrames = sf_read_float(file, &tempData[0], tempData.size());
          for (int i = 0; i < numberOfFrames / info.channels; ++i) {
            data[i] = tempData[i * info.channels];
          }
        } else {
          numberOfFrames = sf_read_float(file, data, capacity);
        }

        CHECK_EQ(numberOfFrames / info.channels, capacity) <<
          "File could not fill provided array";

        status = sf_close(file);
        CHECK_EQ(status, 0) << "Failed to close file: ''" << filePath << "': "
          << sf_strerror(file);

        return numberOfFrames;
    }

    int ReadAudioFile(const std::string& filePath, double* data, int capacity,
        int offset) {
        SF_INFO info = SF_INFO();

        SNDFILE* file = sf_open(filePath.c_str(), SFM_READ, &info);\
        CHECK_EQ(sf_error(file), SF_ERR_NO_ERROR) << "Can't open file '" <<
          filePath << "': " << sf_strerror(file);

        sf_count_t status = sf_seek(file, offset, SEEK_SET);
        CHECK_NE(status, -1) << "Can't seek to offset in: '" << filePath <<
          "': " << sf_strerror(file);

        sf_count_t numberOfFrames;
        if (info.channels != 1) {
          //  Non-mono audio files will only have first channel read
          std::valarray<double> tempData(info.channels * capacity);
          numberOfFrames = sf_read_double(file, &tempData[0], tempData.size());
          for (int i = 0; i < numberOfFrames / info.channels; ++i) {
            data[i] = tempData[i * info.channels];
          }
        } else {
          numberOfFrames = sf_read_double(file, data, capacity);
        }

        CHECK_EQ(numberOfFrames / info.channels, capacity) <<
          "File could not fill provided array";

        status = sf_close(file);
        CHECK_EQ(status, 0) << "Failed to close file: ''" << filePath << "': "
          << sf_strerror(file);

        return numberOfFrames;
    }

}  // namespace caffe
#endif  // USE_AUDIO
