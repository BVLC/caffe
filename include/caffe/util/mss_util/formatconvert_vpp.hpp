
#ifndef INCLUDE_CAFFE_UTIL_MSS_UTIL_FORMATCONVERT_VPP_HPP_
#define INCLUDE_CAFFE_UTIL_MSS_UTIL_FORMATCONVERT_VPP_HPP_

#ifdef USE_FFMPEG_QSV

#include "caffe/util/mss_util/base_vpp.hpp"

namespace caffe { namespace mss {

class FormatConvertElement : public BaseVPPElement {
public:
    FormatConvertElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator);
    virtual ~FormatConvertElement();
protected:
    virtual void initPrivateParam(mfxFrameSurface1 *msdk_surface);
    virtual void initPrivateParamRelease();
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV

#endif /* INCLUDE_CAFFE_UTIL_MSS_UTIL_FORMATCONVERT_VPP_HPP_ */
