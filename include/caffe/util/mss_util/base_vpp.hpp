
#ifndef INCLUDE_CAFFE_UTIL_MSS_UTIL_BASE_VPP_HPP_
#define INCLUDE_CAFFE_UTIL_MSS_UTIL_BASE_VPP_HPP_

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

namespace caffe { namespace mss {

#define MSDK_ALIGN32(X)     (((mfxU32)((X)+31)) & (~ (mfxU32)31))
#define MSDK_ALIGN16(X)     (((X + 15) >> 4) << 4)

class BaseVPPElement {
public:
    BaseVPPElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator);
    virtual ~BaseVPPElement();
    void setFourCC(const mfxU32 fourcc) { m_out_fourcc = fourcc; }
    void init(mfxFrameSurface1 *msdk_surface, const mfxU16 out_w, const mfxU16 out_h);
    mfxFrameSurface1* process(AVFrame *decode_frame);
    int getOutWidth();
    int getOutHeight();
protected:
    mfxStatus allocFrames(mfxFrameAllocRequest *pRequest, bool isVppIn);
    int getFreeSurfaceIndex(bool isVppIn);
    virtual void initPrivateParam(mfxFrameSurface1 *msdk_surface) = 0;
    virtual void initPrivateParamRelease() = 0;
    int pixfmtTOmfxfourcc(int format);
    void writeSurfaceData(AVFrame *decode_frame, mfxFrameSurface1 *msdk_surface);
protected:
    MFXVideoSession *m_session;
    SurfaceAllocator *m_pMFXAllocator;
    mfxU32 m_out_fourcc;
    mfxFrameSurface1 **m_surface_pool_in;
    mfxFrameSurface1 **m_surface_pool_out;
    unsigned m_num_of_surf_in;
    unsigned m_num_of_surf_out;
    mfxFrameAllocResponse m_mfxResponse_in;
    mfxFrameAllocResponse m_mfxResponse_out;
    mfxVideoParam m_video_param;
    mfxExtBuffer *pExtBuf[1];
    MFXVideoVPP *m_mfx_vpp;
    mfxU16 m_in_w;
    mfxU16 m_in_h;
    bool m_initialized;
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV

#endif /* INCLUDE_CAFFE_UTIL_MSS_UTIL_BASE_VPP_HPP_ */
