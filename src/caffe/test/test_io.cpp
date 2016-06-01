#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class IOTest : public ::testing::Test {};

bool ReadImageToDatumReference(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

TEST_F(IOTest, TestReadImageToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumReference) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 0, 0, true, &datum);
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}


TEST_F(IOTest, TestReadImageToDatumReferenceResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 100, 200, true, &datum);
  ReadImageToDatumReference(filename, 0, 100, 200, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadImageToDatumContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(data[index++] ==
          static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumContentGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int h = 0; h < datum.height(); ++h) {
    for (int w = 0; w < datum.width(); ++w) {
      EXPECT_TRUE(data[index++] == static_cast<char>(cv_img.at<uchar>(h, w)));
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 100, 200, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 100);
  EXPECT_EQ(datum.width(), 200);
}

TEST_F(IOTest, TestReadImageToDatumResizedMinDim) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 0, 0, 200, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);

  ReadImageToDatum(filename, 0, 0, 0, 720, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 720);
  EXPECT_EQ(datum.width(), 960);

  ReadImageToDatum(filename, 0, 0, 0, 648, 720, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 540);
  EXPECT_EQ(datum.width(), 720);
}

TEST_F(IOTest, TestReadImageToDatumResizedMaxDim) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 0, 0, 0, 500, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);

  ReadImageToDatum(filename, 0, 0, 0, 0, 240, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 180);
  EXPECT_EQ(datum.width(), 240);

  ReadImageToDatum(filename, 0, 0, 0, 540, 960, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 540);
  EXPECT_EQ(datum.width(), 720);
}

TEST_F(IOTest, TestReadImageToDatumResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 256, 256, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToDatumGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, 256, 256, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToCVMat) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 100, 200);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 100);
  EXPECT_EQ(cv_img.cols, 200);
}

TEST_F(IOTest, TestReadImageToCVMatResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestReadImageToCVMatResizedMinDim) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 0, 0, 200, 0, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);

  cv_img = ReadImageToCVMat(filename, 0, 0, 720, 0, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 720);
  EXPECT_EQ(cv_img.cols, 960);

  cv_img = ReadImageToCVMat(filename, 0, 0, 648, 720, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 540);
  EXPECT_EQ(cv_img.cols, 720);
}

TEST_F(IOTest, TestReadImageToCVMatResizedMaxDim) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 0, 0, 0, 500, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);

  cv_img = ReadImageToCVMat(filename, 0, 0, 0, 240, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 180);
  EXPECT_EQ(cv_img.cols, 240);

  cv_img = ReadImageToCVMat(filename, 0, 0, 540, 960, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 540);
  EXPECT_EQ(cv_img.cols, 720);
}

TEST_F(IOTest, TestReadImageToCVMatGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestCVMatToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestCVMatToDatumContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatum(filename, 0, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestCVMatToDatumReference) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadFileToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(datum.encoded());
  EXPECT_EQ(datum.label(), -1);
  EXPECT_EQ(datum.data().size(), 140391);
}

TEST_F(IOTest, TestDecodeDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatum(&datum, true));
  EXPECT_FALSE(DecodeDatum(&datum, true));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMat) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
  cv_img = DecodeDatumToCVMat(datum, false);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

TEST_F(IOTest, TestDecodeDatumNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumNativeGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat_gray.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, false, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNativeGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat_gray.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContentNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
