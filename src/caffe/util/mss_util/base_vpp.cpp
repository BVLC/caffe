
#ifdef USE_FFMPEG_QSV

#include <unistd.h>
#include "caffe/util/mss_util/base_vpp.hpp"

namespace caffe { namespace mss {

BaseVPPElement::BaseVPPElement(MFXVideoSession *session, SurfaceAllocator *pMFXAllocator) {
    m_out_fourcc = MFX_FOURCC_RGB4;
    m_session = session;
    m_pMFXAllocator = pMFXAllocator;
    memset(&m_video_param, 0, sizeof(m_video_param));
    memset(&m_mfxResponse_in, 0, sizeof(m_mfxResponse_in));
    memset(&m_mfxResponse_out, 0, sizeof(m_mfxResponse_out));
    m_num_of_surf_in = 0;
    m_num_of_surf_out = 0;
    m_mfx_vpp = 0;
    m_in_w = 0;
    m_in_h = 0;
    m_surface_pool_in = 0;
    m_surface_pool_out = 0;
    m_initialized = false;
}


BaseVPPElement::~BaseVPPElement() {
    if (m_mfx_vpp) {
        m_mfx_vpp->Close();
        delete m_mfx_vpp;
        m_mfx_vpp = 0;
    }
    if (m_surface_pool_in) {
        for (unsigned int i = 0; i < m_num_of_surf_in; i++) {
            if (m_surface_pool_in[i]) {
                delete m_surface_pool_in[i];
                m_surface_pool_in[i] = 0;
            }
        }
        delete m_surface_pool_in;
        m_surface_pool_in = NULL;
    }
    if (m_surface_pool_out) {
        for (unsigned int i = 0; i < m_num_of_surf_out; i++) {
            if (m_surface_pool_out[i]) {
                delete m_surface_pool_out[i];
                m_surface_pool_out[i] = 0;
            }
        }
        delete m_surface_pool_out;
        m_surface_pool_out = NULL;
    }
    if (m_pMFXAllocator) {
        m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_mfxResponse_in);
        m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_mfxResponse_out);
        m_pMFXAllocator = NULL;
    }
}

void BaseVPPElement::init(mfxFrameSurface1 *msdk_surface, const mfxU16 out_w, const mfxU16 out_h) {
    mfxStatus sts = MFX_ERR_NONE;
    if (!m_initialized) {
        m_mfx_vpp = new MFXVideoVPP(*m_session);


        memcpy(&m_video_param.vpp.In, &msdk_surface->Info,
                sizeof(m_video_param.vpp.In));

        m_in_w = m_video_param.vpp.In.CropW;
        m_in_h = m_video_param.vpp.In.CropH;
        m_video_param.vpp.Out.FourCC        = m_out_fourcc;
        m_video_param.vpp.Out.ChromaFormat  = m_video_param.vpp.In.ChromaFormat;
        if (m_video_param.vpp.Out.FourCC == MFX_FOURCC_NV12) {
            m_video_param.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
        }
        m_video_param.vpp.Out.PicStruct  = m_video_param.vpp.In.PicStruct;
        m_video_param.vpp.Out.CropX         = 0;
        m_video_param.vpp.Out.CropY         = 0;
        m_video_param.vpp.Out.CropW         = out_w;
        m_video_param.vpp.Out.CropH         = out_h;
        m_video_param.vpp.Out.Width         = MSDK_ALIGN16(m_video_param.vpp.Out.CropW);
        m_video_param.vpp.Out.Height        = (MFX_PICSTRUCT_PROGRESSIVE == m_video_param.vpp.Out.PicStruct) ?
                                              MSDK_ALIGN16(m_video_param.vpp.Out.CropH) : MSDK_ALIGN32(m_video_param.vpp.Out.CropH);
        m_video_param.vpp.Out.FrameRateExtN = m_video_param.vpp.In.FrameRateExtN;
        m_video_param.vpp.Out.FrameRateExtD = m_video_param.vpp.In.FrameRateExtD;
        m_video_param.AsyncDepth = 1;
        m_video_param.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

        initPrivateParam(msdk_surface);
        sts = m_mfx_vpp->Query(&m_video_param, &m_video_param);
        /*
        mfxVideoParam tmpParam={0};
        tmpParam.ExtParam = m_video_param.ExtParam;
        tmpParam.NumExtParam = m_video_param.NumExtParam;
        sts = m_mfx_vpp->Query(&m_video_param, &tmpParam);
        m_video_param=tmpParam;
        */
        mfxFrameAllocRequest VPPRequest[2];
        memset(&VPPRequest, 0, sizeof(mfxFrameAllocRequest) * 2);
        sts = m_mfx_vpp->QueryIOSurf(&m_video_param, VPPRequest);
        if (sts != MFX_ERR_NONE) {
            m_mfx_vpp->Close();
            delete m_mfx_vpp;
            m_mfx_vpp = 0;
            initPrivateParamRelease();
            return;
        }
        m_num_of_surf_in = VPPRequest[0].NumFrameSuggested + 5;
        sts = allocFrames(&VPPRequest[0], true);
        if (sts != MFX_ERR_NONE) {
            m_mfx_vpp->Close();
            delete m_mfx_vpp;
            m_mfx_vpp = 0;
            initPrivateParamRelease();
            return;
        }

        m_num_of_surf_out = VPPRequest[1].NumFrameSuggested + 5;
        sts = allocFrames(&VPPRequest[1], false);
        if (sts != MFX_ERR_NONE) {
            m_mfx_vpp->Close();
            delete m_mfx_vpp;
            m_mfx_vpp = 0;
            initPrivateParamRelease();
            return;
        }

        sts = m_mfx_vpp->Init(&m_video_param);
        initPrivateParamRelease();
        if (sts != MFX_ERR_NONE) {
            m_mfx_vpp->Close();
            delete m_mfx_vpp;
            m_mfx_vpp = 0;
            return;
        }
        m_initialized = true;
    }
}

mfxFrameSurface1* BaseVPPElement::process(AVFrame *decode_frame) {
    mfxFrameSurface1* out = 0;
    mfxStatus sts = MFX_ERR_NONE;
    mfxSyncPoint syncpV;
    int nIndex = MFX_ERR_NOT_FOUND;
    int findCnt = 0;
    if (!m_initialized) {
        return out;
    }

    mfxFrameSurface1* in = m_surface_pool_in[0];
    writeSurfaceData(decode_frame, in);
    while (true) {
        nIndex = getFreeSurfaceIndex(false);
        if (nIndex != MFX_ERR_NOT_FOUND) {
            break;
        } else {
            if (findCnt > 100) {
                return 0;
            }
            usleep(10000);
            findCnt++;
        }
    }
    out = m_surface_pool_out[nIndex];
    sts = m_mfx_vpp->RunFrameVPPAsync(in, out, NULL, &syncpV);
    if (MFX_ERR_NONE < sts && syncpV) {
        sts = MFX_ERR_NONE;
    }
    if (sts != MFX_ERR_NONE) {
        printf ("process surface failed!!!\n");
        return 0;
    }
    m_session->SyncOperation(syncpV, 60000);
    return out;
}

int BaseVPPElement::getOutWidth() {
    int w = 0;
    if (m_initialized) {
        w = m_video_param.vpp.Out.CropW;
    }
    return w;
}
int BaseVPPElement::getOutHeight() {
    int h = 0;
    if (m_initialized) {
        h = m_video_param.vpp.Out.CropH;
    }
    return h;
}

mfxStatus BaseVPPElement::allocFrames(mfxFrameAllocRequest *pRequest, bool isVppIn) {
    mfxStatus sts = MFX_ERR_NONE;
    mfxU16 i;
    mfxFrameAllocResponse *pResponse = 0;
    mfxFrameSurface1 ***surface_pool = 0;
    unsigned num_of_surf = 0;
    if (isVppIn) {
        surface_pool = &m_surface_pool_in;
        pRequest->NumFrameMin = m_num_of_surf_in;
        pRequest->NumFrameSuggested = m_num_of_surf_in;
        num_of_surf = m_num_of_surf_in;
        pResponse = &m_mfxResponse_in;
    } else {
        surface_pool = &m_surface_pool_out;
        pRequest->NumFrameMin = m_num_of_surf_out;
        pRequest->NumFrameSuggested = m_num_of_surf_out;
        num_of_surf = m_num_of_surf_out;
        pResponse = &m_mfxResponse_out;
    }
    sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, pRequest, pResponse);

    if (sts != MFX_ERR_NONE) {
        printf("AllocFrame failed %d\n", sts);
        return sts;
    }
    (*surface_pool) = new mfxFrameSurface1*[num_of_surf];
    for (i = 0; i < num_of_surf; i++) {
        (*surface_pool)[i] = new mfxFrameSurface1;
        memset((*surface_pool)[i], 0, sizeof(mfxFrameSurface1));
        memcpy(&((*surface_pool)[i]->Info), &(pRequest->Info), sizeof((*surface_pool)[i]->Info));
        (*surface_pool)[i]->Data.MemId = pResponse->mids[i];
    }
    return MFX_ERR_NONE;
}

int BaseVPPElement::getFreeSurfaceIndex(bool isVppIn) {
    mfxFrameSurface1 **surface_pool = 0;
    unsigned num_of_surf = 0;
    if (isVppIn) {
        surface_pool = m_surface_pool_in;
        num_of_surf = m_num_of_surf_in;
    } else {
        surface_pool = m_surface_pool_out;
        num_of_surf = m_num_of_surf_out;
    }
    if (surface_pool) {
        for (mfxU16 i = 0; i < num_of_surf; i++) {
            if (0 == surface_pool[i]->Data.Locked) {
                return i;
            }
        }
    }

    return MFX_ERR_NOT_FOUND;
}

int BaseVPPElement::pixfmtTOmfxfourcc(int format) {
    switch(format) {
        case AV_PIX_FMT_YUV420P:
            return MFX_FOURCC_YV12;
        case AV_PIX_FMT_NV12:
            return MFX_FOURCC_NV12;
        case AV_PIX_FMT_YUYV422:
            return MFX_FOURCC_YUY2;
        case AV_PIX_FMT_RGB32:
            return MFX_FOURCC_RGB4;
    }
    return MFX_FOURCC_NV12;
}

void BaseVPPElement::writeSurfaceData(AVFrame *decode_frame, mfxFrameSurface1 *msdk_surface) {
    //int mfx_fourcc = pixfmtTOmfxfourcc(decode_frame->format);
    mfxU32 w, h, i, pitch;
    mfxU8 *ptr;
    mfxFrameData* pData = &(msdk_surface->Data);
    mfxFrameInfo* pInfo = &(msdk_surface->Info);

    if (pInfo->CropH > 0 && pInfo->CropW > 0) {
        w = pInfo->CropW;
        h = pInfo->CropH;
    } else {
        w = pInfo->Width;
        h = pInfo->Height;
    }

    pData->Pitch = decode_frame->linesize[0];
    pitch = pData->Pitch;
    m_pMFXAllocator->vaapiLockFrame(msdk_surface->Data.MemId, &msdk_surface->Data);
    if(pInfo->FourCC == MFX_FOURCC_YV12) {
        ptr = pData->Y + pInfo->CropX + pInfo->CropY * pitch;

        mfxU8 * src_data = decode_frame->data[0];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w);
            ptr += pitch;
            src_data += w;
        }
        w     >>= 1;
        h     >>= 1;
        pitch >>= 1;
        // load U/V
        ptr = (pData->V) + (pInfo->CropX >> 1) + (pInfo->CropY >> 1) * pitch;
        src_data = decode_frame->data[2];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w);
            ptr += pitch;
            src_data += w;
        }
        // load V/U
        ptr  = (pData->U) + (pInfo->CropX >> 1) + (pInfo->CropY >> 1) * pitch;
        src_data = decode_frame->data[1];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w);
            ptr += pitch;
            src_data += w;
        }

    }
    else if( pInfo->FourCC == MFX_FOURCC_NV12 ) {
        ptr = pData->Y + pInfo->CropX + pInfo->CropY * pitch;

        // read luminance plane
        mfxU8 * src_data = decode_frame->data[0];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w);
            ptr += pitch;
            src_data += w;
        }
        // load UV
        h     >>= 1;
        ptr = pData->UV + pInfo->CropX + (pInfo->CropY >> 1) * pitch;
        src_data = decode_frame->data[1];
        for (i = 0; i < h; i++) {
            memcpy(ptr, src_data, w);
            ptr += pitch;
            src_data += w;
        }
    } else if (pInfo->FourCC == MFX_FOURCC_RGB4) {
        ptr = std::min(std::min(pData->R, pData->G), pData->B );
        ptr = ptr + pInfo->CropX + pInfo->CropY * pitch;

        mfxU8 * src_data = decode_frame->data[0];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w * 4);
            ptr += pitch;
            src_data += 4*w;
        }
    } else if (pInfo->FourCC == MFX_FOURCC_YUY2) {
        ptr = pData->Y + pInfo->CropX + pInfo->CropY * pitch;

        mfxU8 * src_data = decode_frame->data[0];
        for(i = 0; i < h; i++) {
            memcpy(ptr, src_data, w * 2);
            ptr += pitch;
            src_data += 2*w;
        }

    }
    m_pMFXAllocator->vaapiUnlockFrame(msdk_surface->Data.MemId, &msdk_surface->Data);
}

}
}

#endif
