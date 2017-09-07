
#ifdef USE_FFMPEG_QSV

#include "caffe/util/mss_util/formatconvert_vpp.hpp"

namespace caffe { namespace mss {

FormatConvertElement::FormatConvertElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator) : BaseVPPElement(session, pMFXAllocator) {

}

FormatConvertElement::~FormatConvertElement() {

}

void FormatConvertElement::initPrivateParam(mfxFrameSurface1 *msdk_surface) {
    m_video_param.vpp.Out.CropX         = m_video_param.vpp.In.CropX;
    m_video_param.vpp.Out.CropY         = m_video_param.vpp.In.CropY;
    m_video_param.vpp.Out.CropW         = m_video_param.vpp.In.CropW;
    m_video_param.vpp.Out.CropH         = m_video_param.vpp.In.CropH;
    m_video_param.vpp.Out.Width         = MSDK_ALIGN16(m_video_param.vpp.Out.CropW);
    m_video_param.vpp.Out.Height        = (MFX_PICSTRUCT_PROGRESSIVE == m_video_param.vpp.Out.PicStruct) ?
                                          MSDK_ALIGN16(m_video_param.vpp.Out.CropH) : MSDK_ALIGN32(m_video_param.vpp.Out.CropH);
}

void FormatConvertElement::initPrivateParamRelease() {

}

}
}

#endif
