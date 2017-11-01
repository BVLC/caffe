
#ifndef INCLUDE_CAFFE_UTIL_MSS_UTIL_PIPELINE_HPP_
#define INCLUDE_CAFFE_UTIL_MSS_UTIL_PIPELINE_HPP_

#ifdef USE_FFMPEG_QSV

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

#include "mfxvideo++.h"
#include "caffe/util/mss_util/surface_allocator.hpp"
#include "caffe/util/mss_util/base_vpp.hpp"


namespace caffe { namespace mss {

typedef struct Image_FFMPEG_
{
    unsigned char* data;
    int step;
    int width;
    int height;
    int cn;
} Image_FFMPEG;

class Pipeline {
public:
    Pipeline();
    virtual ~Pipeline();
    void initPipeline(AVFrame *decode_frame, int out_w, int out_, int out_layout, AVQSVDeviceContext *hwctx);
    bool isInitialized();
    void processFrame(AVFrame *decode_frame, Image_FFMPEG* layout_data, Image_FFMPEG* org_data);
private:
    void initVaDisp();
    void destroyVaDisp();
    void initVideoSession();
    void createAllocator();
    void copyRGBData(mfxFrameSurface1* layout, mfxFrameSurface1* org);
    mfxStatus writeInputBuff(unsigned char** outData, mfxFrameSurface1* pSurface, bool isCrop);
private:
    MFXVideoSession m_layout_session;
    MFXVideoSession m_rgb4_session;
    int m_dri_fd;
    VADisplay m_va_dpy;
    SurfaceAllocator *m_layout_allocator;
    SurfaceAllocator *m_rgb4_allocator;
    BaseVPPElement *m_layout_vpp;
    BaseVPPElement *m_rgb4_vpp;
    bool m_initialized;
    unsigned char* m_layout_data;
    unsigned char* m_rgb4_data;
    int m_layout;
    int m_out_w;
    int m_out_h;
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV

#endif /* INCLUDE_CAFFE_UTIL_MSS_UTIL_PIPELINE_HPP_ */
