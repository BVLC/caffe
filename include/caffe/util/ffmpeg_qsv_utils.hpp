
#ifndef INCLUDE_CAFFE_UTIL_FFMPEG_QSV_UTILS_HPP_
#define INCLUDE_CAFFE_UTIL_FFMPEG_QSV_UTILS_HPP_

#ifdef USE_FFMPEG_QSV

#include <opencv2/core/core.hpp>

#include "caffe/util/mss_util/pipeline.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfiltergraph.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_qsv.h>
#include <libavutil/avstring.h>

#ifdef __cplusplus
}
#endif

using caffe::mss::Pipeline;
using caffe::mss::Image_FFMPEG;

namespace caffe { namespace qsv {

typedef struct FilteringContext {
    AVCodecContext  *dec_ctx;
    AVFilterContext *buffersink_ctx;
    AVFilterContext *buffersrc_ctx;
    AVFilterGraph *filter_graph;
    int initiallized;
} FilteringContext;

class QsvVideoReader {

public :
    QsvVideoReader();
    virtual ~QsvVideoReader();
    
    bool open( const char* _filename );
    void close();
    bool isOpened();
    void seek(int64_t _frame_number);
    int64_t get_total_frames();
    void getVideoFrame(cv::Mat &cv_img, cv::Mat &org_img);
    void setResize(int width, int height, int layout);
private:
    void init();
    double get_duration_sec();
    double r2d(AVRational r);
    double get_fps();
    bool decodeFrame();
    double dts_to_sec(int64_t dts);
    int64_t dts_to_frame_number(int64_t dts);
    //int init_filter(FilteringContext* fctx, AVCodecContext *dec_ctx, const char *filter_spec);
    bool init_swscontext(enum AVPixelFormat dec_pix_fmts);
private:
    AVFormatContext * ic;
    int               video_stream;
    int64_t           picture_pts;

    AVPacket          packet;
    Image_FFMPEG      frame;
    Image_FFMPEG      org_frame;
    struct SwsContext *img_convert_ctx;
    AVFrame           rgb_picture;

    int64_t frame_number, first_frame_number;

    double eps_zero;
    char              * filename;
    AVDictionary *dict;
    FilteringContext filter_ctx;
    int out_width;
    int out_height;
    int out_layout;
    Pipeline          mss_pipeline;
    bool              use_hw_decoder;
};

}
}

#endif

#endif /* INCLUDE_CAFFE_UTIL_FFMPEG_QSV_UTILS_HPP_ */
