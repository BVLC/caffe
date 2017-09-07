
#ifndef INCLUDE_CAFFE_UTIL_MSS_UTIL_LAYOUT_VPP_HPP_
#define INCLUDE_CAFFE_UTIL_MSS_UTIL_LAYOUT_VPP_HPP_

#ifdef USE_FFMPEG_QSV

#include "caffe/util/mss_util/base_vpp.hpp"

namespace caffe { namespace mss {

typedef enum {
    WARP = 1,
    FIT_SMALL_SIZE = 2,
    FIT_LARGE_SIZE_AND_PAD = 3
} VPP_LAYOUT_TYPE;

class LayoutVPPElement : public BaseVPPElement {
public:
    LayoutVPPElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator, VPP_LAYOUT_TYPE type);
    virtual ~LayoutVPPElement();

protected:
    virtual void initPrivateParam(mfxFrameSurface1 *msdk_surface);
    virtual void initPrivateParamRelease();
private:
    void setCropLayoutParam();
    void setPadLayoutParam();
    void computePosition(mfxVPPCompInputStream &inputStream);
private:
    VPP_LAYOUT_TYPE m_layout_type;
    mfxExtVPPComposite m_layout_param;
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV

#endif /* INCLUDE_CAFFE_UTIL_MSS_UTIL_LAYOUT_VPP_HPP_ */
