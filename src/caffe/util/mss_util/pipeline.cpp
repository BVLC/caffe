
#ifdef USE_FFMPEG_QSV

#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <memory.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_x11.h>

#include "caffe/util/mss_util/pipeline.hpp"
#include "caffe/util/mss_util/layout_vpp.hpp"
#include "caffe/util/mss_util/formatconvert_vpp.hpp"

namespace caffe { namespace mss {

#define MSDK_MIN(A, B)                           (((A) < (B)) ? (A) : (B))

Pipeline::Pipeline() {
    m_dri_fd = -1;
    m_va_dpy = 0;
    m_layout_allocator = 0;
    m_rgb4_allocator = 0;
    m_layout_vpp = 0;
    m_rgb4_vpp = 0;
    m_initialized = false;
#ifdef CREATE_NEW_DISPLAY
    initVaDisp();
#endif
    m_layout_data = 0;
    m_rgb4_data = 0;
    m_layout = 1;
    m_out_w = 0;
    m_out_h = 0;
}


Pipeline::~Pipeline() {
    if (m_layout_vpp) {
        delete m_layout_vpp;
        m_layout_vpp = 0;
    }
    if (m_rgb4_vpp) {
        delete m_rgb4_vpp;
        m_rgb4_vpp = 0;
    }
#ifdef CREATE_NEW_DISPLAY
    destroyVaDisp();
#endif
    if (m_layout_allocator) {
        delete m_layout_allocator;
        m_layout_allocator = 0;
    }
    if (m_rgb4_allocator) {
        delete m_rgb4_allocator;
        m_rgb4_allocator = 0;
    }
    if (m_layout_data) {
        delete m_layout_data;
        m_layout_data = 0;
    }
    if (m_rgb4_data) {
        delete m_rgb4_data;
        m_rgb4_data = 0;
    }
}

void Pipeline::initPipeline(AVFrame *decode_frame, int out_w, int out_h, int out_layout, AVQSVDeviceContext *hwctx) {
#ifndef CREATE_NEW_DISPLAY
    int ret = MFXVideoCORE_GetHandle(hwctx->session, MFX_HANDLE_VA_DISPLAY, &m_va_dpy);
#endif
    mfxFrameSurface1* surface = (mfxFrameSurface1*)decode_frame->data[3];
    if (!surface) {
        printf("input surface is null\n");
        return;
    }
    int in_w = surface->Info.CropW;
    int in_h = surface->Info.CropH;
    if (in_w <= 0 || in_h <= 0) {
        printf("width or height invalid\n");
        return ;
    }
    initVideoSession();
    createAllocator();
    m_layout_vpp = new LayoutVPPElement(&m_layout_session, m_layout_allocator, (VPP_LAYOUT_TYPE)out_layout);
    m_rgb4_vpp = new FormatConvertElement(&m_rgb4_session, m_rgb4_allocator);
    if (!m_layout_vpp || !m_rgb4_vpp) {
        printf("vpp create failed\n");
        return;
    }
    m_layout_vpp->init(surface, out_w, out_h);
    m_rgb4_vpp->init(surface, in_w, in_h);
    int layout_len = m_layout_vpp->getOutWidth()*m_layout_vpp->getOutHeight()*3;
    m_layout_data = new unsigned char[layout_len];
    int org_len = m_rgb4_vpp->getOutWidth()*m_rgb4_vpp->getOutHeight()*3;
    m_rgb4_data = new unsigned char[org_len];
    if (!m_layout_data || !m_rgb4_data) {
        return;
    }
    m_layout = out_layout;
    m_out_w = m_layout_vpp->getOutWidth();
    m_out_h = m_layout_vpp->getOutHeight();
    m_initialized = true;
}

bool Pipeline::isInitialized() {
    return m_initialized;
}

void Pipeline::processFrame(AVFrame *decode_frame, Image_FFMPEG* layout_data, Image_FFMPEG* org_data) {
    mfxFrameSurface1* layout = m_layout_vpp->process(decode_frame);
    mfxFrameSurface1* rgb = m_rgb4_vpp->process(decode_frame);
    if (!layout || !rgb) {
        printf("vpp process failed!\n");
        return;
    }
    copyRGBData(layout, rgb);
    layout_data->data = m_layout_data;
    layout_data->step = m_out_w*3;
    layout_data->cn = 3;
    layout_data->width = m_out_w;
    layout_data->height = m_out_h;
    org_data->data = m_rgb4_data;
}

void Pipeline::initVaDisp() {

    int major_version, minor_version;
    int ret = VA_STATUS_SUCCESS;

    m_dri_fd = open("/dev/dri/renderD128", O_RDWR);
    if (-1 == m_dri_fd) {
        printf("Open dri renderD128 failed!\n");
        return;
    }

    m_va_dpy = vaGetDisplayDRM(m_dri_fd);
    ret = vaInitialize(m_va_dpy, &major_version, &minor_version);
    if (VA_STATUS_SUCCESS != ret) {
        //if failed, try to use card0
        if (NULL != m_va_dpy) {
            vaTerminate(m_va_dpy);
            m_va_dpy = NULL;
        }
        if (m_dri_fd) {
            close(m_dri_fd);
            m_dri_fd = -1;
        }

        m_dri_fd = open("/dev/dri/card0", O_RDWR);
        if (-1 == m_dri_fd) {
            printf("Open dri/card0 failed!\n");
            return;
        }
        m_va_dpy = vaGetDisplayDRM(m_dri_fd);
        ret = vaInitialize(m_va_dpy, &major_version, &minor_version);
        if (VA_STATUS_SUCCESS != ret) {
            printf("vaInitialize failed\n");
                return;
        }
    }
}

void Pipeline::destroyVaDisp() {
    if (m_va_dpy) {
        vaTerminate(m_va_dpy);
        m_va_dpy = 0;
    }
    if (m_dri_fd > 0) {
        close(m_dri_fd);
        m_dri_fd = -1;
    }
}

void Pipeline::initVideoSession() {
    mfxStatus sts = MFX_ERR_NONE;
    mfxVersion version = {10, 1};
    mfxIMPL    impl    = MFX_IMPL_VIA_VAAPI | MFX_IMPL_HARDWARE;
    sts = m_layout_session.Init(impl, &version);
    if (sts != MFX_ERR_NONE) {
        printf("MSS MFX layout video session init fail!!");
    }
    sts = m_layout_session.SetHandle((mfxHandleType)MFX_HANDLE_VA_DISPLAY, m_va_dpy);
    if (sts != MFX_ERR_NONE) {
        printf("MSS MFX layout video session SetHandle fail!!");
    }
    //sts = m_rgb4_session.InitEx(initPar);
    sts = m_rgb4_session.Init(impl, &version);
    if (sts != MFX_ERR_NONE) {
        printf("MSS MFX resize video session init fail!!");
    }
    sts = m_rgb4_session.SetHandle((mfxHandleType)MFX_HANDLE_VA_DISPLAY, m_va_dpy);
    if (sts != MFX_ERR_NONE) {
        printf("MSS MFX resize video session SetHandle fail!!");
    }
    m_layout_session.JoinSession(m_rgb4_session);
}

void Pipeline::createAllocator() {
    mfxStatus sts = MFX_ERR_NONE;
    if (!m_layout_allocator) {
        m_layout_allocator = new SurfaceAllocator;
    }
    if (!m_rgb4_allocator) {
        m_rgb4_allocator = new SurfaceAllocator;
    }
    if (m_layout_allocator && m_rgb4_allocator) {
        sts = m_layout_allocator->init(&(m_va_dpy));
        if (sts == MFX_ERR_NONE) {
            sts = m_layout_session.SetFrameAllocator(m_layout_allocator);
        }
        sts = m_rgb4_allocator->init(&(m_va_dpy));
        if (sts == MFX_ERR_NONE) {
            sts = m_rgb4_session.SetFrameAllocator(m_rgb4_allocator);
        }
    } else {
        printf("Allocator failed!!!\n");
    }
}

void Pipeline::copyRGBData(mfxFrameSurface1* layout, mfxFrameSurface1* org) {
    mfxStatus sts = MFX_ERR_NONE;
    m_layout_allocator->vaapiLockFrame(layout->Data.MemId, &layout->Data);
    sts = writeInputBuff(&m_layout_data, layout, true);
    //printf("writeInputBuff layout sts [%d]!!!\n", sts);
    m_layout_allocator->vaapiUnlockFrame(layout->Data.MemId, &layout->Data);

    m_rgb4_allocator->vaapiLockFrame(org->Data.MemId, &org->Data);
    sts = writeInputBuff(&m_rgb4_data, org, false);
    //printf("writeInputBuff org sts [%d]!!!\n", sts);
    m_rgb4_allocator->vaapiUnlockFrame(org->Data.MemId, &org->Data);
}

mfxStatus Pipeline::writeInputBuff(unsigned char** outData, mfxFrameSurface1* pSurface, bool isCrop) {
    mfxStatus sts = MFX_ERR_NONE;
    mfxFrameInfo &pInfo = pSurface->Info;
    mfxFrameData &pData = pSurface->Data;

    mfxU32 i, j, h, w;
    mfxU16 cropX, cropY;
    int buflen;
    unsigned char *tmpBuf = NULL;
    switch (pInfo.FourCC)
    {
    case MFX_FOURCC_RGB4:
        mfxU8* ptr;

        if (pInfo.CropH > 0 && pInfo.CropW > 0) {
            w = pInfo.CropW;
            h = pInfo.CropH;
        } else {
            w = pInfo.Width;
            h = pInfo.Height;
        }
        cropX = pInfo.CropX;
        cropY = pInfo.CropY;
        /*
        if (m_layout == 2 && isCrop) {
            if (w > m_out_w) {
                cropX = (w - m_out_w)/2;
                w = m_out_w;
            }
            if (h > m_out_h) {
                cropY = (h - m_out_h)/2;
                h = m_out_h;
            }
        }
        */
        ptr = MSDK_MIN( MSDK_MIN(pData.R, pData.G), pData.B);
        //ptr = ptr + pInfo.CropX + pInfo.CropY * pData.Pitch;
        ptr = ptr + cropX + cropY * pData.Pitch;
        buflen = w*4;
        tmpBuf = new unsigned char[buflen];
        for(i = 0; i < h; i++)
        {
            memcpy(tmpBuf, ptr + i * pData.Pitch, 4*w);
            for (j = 0; j < w; j ++) {
                outData[0][i*w*3 + j*3] = tmpBuf[j*4];
                outData[0][i*w*3 + j*3 + 1] = tmpBuf[j*4 + 1];
                outData[0][i*w*3 + j*3 + 2] = tmpBuf[j*4 + 2];
            }
        }
        delete [] tmpBuf;
        break;
    default:
        return MFX_ERR_UNSUPPORTED;
    }
    return sts;
}

}
}

#endif
