
#ifdef USE_FFMPEG_QSV

#include <math.h>
#include "caffe/util/mss_util/layout_vpp.hpp"

namespace caffe { namespace mss {

LayoutVPPElement::LayoutVPPElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator, VPP_LAYOUT_TYPE type) : BaseVPPElement(session, pMFXAllocator) {
    m_layout_type = type;
    memset(&m_layout_param, 0, sizeof(mfxExtVPPComposite));
}

LayoutVPPElement::~LayoutVPPElement() {

}

void LayoutVPPElement::initPrivateParam(mfxFrameSurface1 *msdk_surface) {
    if (WARP == m_layout_type) {
        printf("only scale frame!!\n");
    } else if (FIT_SMALL_SIZE == m_layout_type) {
        setCropLayoutParam();
    } else if (FIT_LARGE_SIZE_AND_PAD == m_layout_type) {
        setPadLayoutParam();
    }
}

void LayoutVPPElement::initPrivateParamRelease() {
    if (m_layout_param.InputStream) {
        delete[] m_layout_param.InputStream;
    }
}

void LayoutVPPElement::setCropLayoutParam() {
    mfxU16 org_width = m_video_param.vpp.In.CropW;
    mfxU16 org_height = m_video_param.vpp.In.CropH;
    mfxU16 width = m_video_param.vpp.Out.CropW;
    mfxU16 height = m_video_param.vpp.Out.CropH;
    if (org_width == 0 || org_height == 0 || width == 0 || height == 0 || (float)org_width/org_height == (float)width/height) {
        return;
    }
    float width_ratio = (float)org_width/width;
    float height_ratio = (float)org_height/height;

    if (width_ratio > height_ratio) {
        mfxU16 cropW = (mfxU16)round(((float)org_width * height / org_height));
        m_video_param.vpp.Out.CropW = cropW;
    } else {
        mfxU16 cropH = (mfxU16)round(((float)org_height * width / org_width));
        m_video_param.vpp.Out.CropH = cropH;
    }
    m_video_param.vpp.Out.Width         = MSDK_ALIGN16(m_video_param.vpp.Out.CropW);
    m_video_param.vpp.Out.Height        = (MFX_PICSTRUCT_PROGRESSIVE == m_video_param.vpp.Out.PicStruct) ?
                                          MSDK_ALIGN16(m_video_param.vpp.Out.CropH) : MSDK_ALIGN32(m_video_param.vpp.Out.CropH);
}

void LayoutVPPElement::setPadLayoutParam() {
    m_layout_param.Header.BufferId = MFX_EXTBUFF_VPP_COMPOSITE;
    m_layout_param.Header.BufferSz = sizeof(mfxExtVPPComposite);
    m_layout_param.Y = 0;
    m_layout_param.U = 0;
    m_layout_param.V = 0;
    m_layout_param.NumInputStream = 1;
    m_layout_param.InputStream = new mfxVPPCompInputStream[m_layout_param.NumInputStream];
    memset(m_layout_param.InputStream, 0 , sizeof(mfxVPPCompInputStream) * m_layout_param.NumInputStream);

    m_video_param.NumExtParam = 1;
    m_video_param.ExtParam = (mfxExtBuffer **)pExtBuf;
    m_video_param.ExtParam[m_video_param.NumExtParam - 1] = (mfxExtBuffer *) & (m_layout_param);
    computePosition(m_layout_param.InputStream[0]);
}

void LayoutVPPElement::computePosition(mfxVPPCompInputStream &inputStream) {
    mfxU16 org_width = m_video_param.vpp.In.CropW;
    mfxU16 org_height = m_video_param.vpp.In.CropH;
    mfxU16 width = m_video_param.vpp.Out.CropW;
    mfxU16 height = m_video_param.vpp.Out.CropH;
    if (org_width == 0 || org_height == 0 || width == 0 || height == 0 || (float)org_width/org_height == (float)width/height || FIT_SMALL_SIZE == m_layout_type) {
        inputStream.DstX = 0;
        inputStream.DstY = 0;
        inputStream.DstW = width;
        inputStream.DstH = height;
        return;
    }
    float width_ratio = (float)org_width/width;
    float height_ratio = (float)org_height/height;
    mfxU16 dst_width = 0;
    mfxU16 dst_height = 0;
    mfxU16 dst_x = 0;
    mfxU16 dst_y = 0;
    if (width_ratio > height_ratio) {
        dst_width = width;
        dst_height = (mfxU16)round(((float)org_height / width_ratio));
        if (height - dst_height > 0) {
            dst_y += (height - dst_height) / 2;
        }
    } else {
        dst_width = (mfxU16)round(((float)org_width / height_ratio));
        dst_height = height;
        if (width - dst_width > 0) {
            dst_x += (width - dst_width) / 2;
        }
    }
    inputStream.DstX = dst_x;
    inputStream.DstY = dst_y;
    inputStream.DstW = dst_width;
    inputStream.DstH = dst_height;
}

}
}

#endif
