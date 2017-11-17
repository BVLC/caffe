
#ifdef USE_FFMPEG_QSV

#include <assert.h>
#include <algorithm>
#include "caffe/util/mss_util/surface_allocator.hpp"

namespace caffe { namespace mss {

static const mfxU32 MFX_MEMTYPE_MASK = MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN | MFX_MEMTYPE_FROM_VPPOUT;

#define SIZE_ALIGN32(v) (((v + 31) >> 5) << 5)

SurfaceAllocator::SurfaceAllocator() {
    m_va_display = NULL;
    pthis = this;
    Alloc = mfxAllocImpl;
    Lock  = mfxLockImpl;
    Free  = mfxFreeImpl;
    Unlock = mfxUnlockImpl;
    GetHDL = mfxGetFrameHDLImpl;
}

SurfaceAllocator::~SurfaceAllocator() {
    close();
}

mfxStatus SurfaceAllocator::init(VADisplay *dpy)
{
    if (NULL == dpy)
        return MFX_ERR_NOT_INITIALIZED;
    m_va_display = *dpy;

    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::close() {
    std::list<mfxFrameAllocResponse> ::iterator itr;
    for (itr = m_surface_pool.begin(); itr!= m_surface_pool.end(); itr++)
    {
        releaseResponse(&*itr);
    }
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiAllocFrames(mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse) {
    mfxStatus sts = MFX_ERR_NONE;
    if (0 == allocRequest || 0 == allocResponse || 0 == allocRequest->NumFrameSuggested)
        return MFX_ERR_MEMORY_ALLOC;

    if (MFX_ERR_NONE != checkRequestType(allocRequest))
        return MFX_ERR_UNSUPPORTED;

    sts = allocImpl(allocRequest, allocResponse);

    if (sts == MFX_ERR_NONE) {
        m_surface_pool.push_back(mfxFrameAllocResponse());
        m_surface_pool.back() = *allocResponse;
    }
    return sts;
}

mfxStatus SurfaceAllocator::vaapiLockFrame(mfxMemId context, mfxFrameData *frameBuff) {
    mfxStatus mfxSts = MFX_ERR_NONE;
    VAStatus  vaSts  = VA_STATUS_SUCCESS;
    vaapiContext* vaapiCtx = (vaapiContext*)context;

    if (!vaapiCtx || !(vaapiCtx->m_va_surface_id)) {
        return MFX_ERR_INVALID_HANDLE;
    }

    mfxU32 mfx_fourcc = vaapiCtx->m_va_fourcc;
    if (MFX_FOURCC_P8 == mfx_fourcc) {
        VACodedBufferSegment *codedBuf;
           vaSts = vaMapBuffer(m_va_display, *(vaapiCtx->m_va_surface_id), (void **)(&codedBuf));
        mfxSts = convertToMfxStatus(vaSts);
        if (MFX_ERR_NONE == mfxSts) {
               frameBuff->Y = (unsigned char*)codedBuf->buf;
        }
    } else {
        vaSts = vaSyncSurface(m_va_display, *(vaapiCtx->m_va_surface_id));
        mfxSts = convertToMfxStatus(vaSts);
        if (MFX_ERR_NONE == mfxSts) {
            vaSts = vaDeriveImage(m_va_display, *(vaapiCtx->m_va_surface_id), &(vaapiCtx->m_va_image));
            mfxSts = convertToMfxStatus(vaSts);
        } else {
            return mfxSts;
        }

        unsigned char *pSysBuffer = 0;
        if (MFX_ERR_NONE == mfxSts) {
            vaSts = vaMapBuffer(m_va_display, vaapiCtx->m_va_image.buf, (void **) &pSysBuffer);
            mfxSts = convertToMfxStatus(vaSts);
        } else {
            return mfxSts;
        }
        if (MFX_ERR_NONE != mfxSts) {
            return mfxSts;
        }
        VAImage image = vaapiCtx->m_va_image;
        if (MFX_FOURCC_NV12 == image.format.fourcc) {
            if (MFX_FOURCC_NV12 == mfx_fourcc) {
                mapNV12Buffer(frameBuff, pSysBuffer, &image);
            } else {
                mfxSts = MFX_ERR_LOCK_MEMORY;
            }
        } else if (VA_FOURCC_YV12 == image.format.fourcc) {
            if (MFX_FOURCC_YV12 == mfx_fourcc) {
                mapYV12Buffer(frameBuff, pSysBuffer, &image);
            } else {
                mfxSts = MFX_ERR_LOCK_MEMORY;
            }
        } else if (VA_FOURCC_YUY2 == image.format.fourcc) {
            if (MFX_FOURCC_YUY2 == mfx_fourcc) {
                mapYUY12Buffer(frameBuff, pSysBuffer, &image);
            } else {
                mfxSts = MFX_ERR_LOCK_MEMORY;
            }
        } else if (VA_FOURCC_ARGB == image.format.fourcc) {
            if (MFX_FOURCC_RGB4 == mfx_fourcc) {
                mapRGB4Buffer(frameBuff, pSysBuffer, &image);
            } else {
                mfxSts = MFX_ERR_LOCK_MEMORY;
            }
        } else if (VA_FOURCC_P208 == image.format.fourcc) {
            if (MFX_FOURCC_NV12 == mfx_fourcc) {
                frameBuff->Pitch = (mfxU16)image.pitches[0];
                frameBuff->Y = pSysBuffer + image.offsets[0];
            } else {
                mfxSts = MFX_ERR_LOCK_MEMORY;
            }
        } else {
            mfxSts = MFX_ERR_LOCK_MEMORY;
        }
    }
    return mfxSts;
}

mfxStatus SurfaceAllocator::vaapiUnlockFrame(mfxMemId context, mfxFrameData *frameBuff) {
    vaapiContext* vaapiCtx = (vaapiContext*)context;

    if (!vaapiCtx || !(vaapiCtx->m_va_surface_id)) {
        return MFX_ERR_INVALID_HANDLE;
    }

    mfxU32 mfx_fourcc = vaapiCtx->m_va_fourcc;

    if (MFX_FOURCC_P8 != mfx_fourcc) {
        vaUnmapBuffer(m_va_display, vaapiCtx->m_va_image.buf);
        vaDestroyImage(m_va_display, vaapiCtx->m_va_image.image_id);

        if (NULL != frameBuff) {
            frameBuff->Pitch = 0;
            frameBuff->A = 0;
            frameBuff->Y = 0;
            frameBuff->U = 0;
            frameBuff->V = 0;
        }
    } else {
        vaUnmapBuffer(m_va_display, *(vaapiCtx->m_va_surface_id));
    }
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiGetFrameHDL(mfxMemId context, mfxHDL *handle) {
    vaapiContext* vaapiCtx = (vaapiContext*)context;

    if (!handle || !vaapiCtx || !(vaapiCtx->m_va_surface_id)) {
        return MFX_ERR_INVALID_HANDLE;
    }

    *handle = vaapiCtx->m_va_surface_id;
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiFreeFrames(mfxFrameAllocResponse *allocResponse) {
    if (allocResponse == 0)
        return MFX_ERR_INVALID_HANDLE;

    mfxStatus sts = MFX_ERR_NONE;

    std::list<mfxFrameAllocResponse>::iterator itr;
    for (itr = m_surface_pool.begin(); itr != m_surface_pool.end(); itr++) {
        if ((*itr).mids != 0 && allocResponse->mids != 0 && (*itr).mids[0] == allocResponse->mids[0] &&
                (*itr).NumFrameActual == allocResponse->NumFrameActual ) {
            sts = releaseResponse(allocResponse);
            m_surface_pool.erase(itr);
            return sts;
        }
    }

    return MFX_ERR_INVALID_HANDLE;
}

mfxStatus SurfaceAllocator::mfxAllocImpl(mfxHDL pthis, mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator *vaapiAlloc = (SurfaceAllocator *)pthis;

    return vaapiAlloc->vaapiAllocFrames(allocRequest, allocResponse);
}

mfxStatus SurfaceAllocator::mfxLockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator *vaapiAlloc = (SurfaceAllocator *)pthis;

    return vaapiAlloc->vaapiLockFrame(mid, ptr);
}

mfxStatus SurfaceAllocator::mfxUnlockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator *vaapiAlloc = (SurfaceAllocator *)pthis;

    return vaapiAlloc->vaapiUnlockFrame(mid, ptr);
}

mfxStatus SurfaceAllocator::mfxGetFrameHDLImpl(mfxHDL pthis, mfxMemId mid, mfxHDL *handle) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator *vaapiAlloc = (SurfaceAllocator *)pthis;

    return vaapiAlloc->vaapiGetFrameHDL(mid, handle);
}

mfxStatus SurfaceAllocator::mfxFreeImpl(mfxHDL pthis, mfxFrameAllocResponse *allocResponse) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator *vaapiAlloc = (SurfaceAllocator *)pthis;

    return vaapiAlloc->vaapiFreeFrames(allocResponse);
}

unsigned int SurfaceAllocator::convertToVAFormat(mfxU32 fourcc)
{
    unsigned int va_fourcc = 0;
    if (MFX_FOURCC_YUY2 == fourcc) {
        va_fourcc = VA_FOURCC_YUY2;
    } else if (MFX_FOURCC_NV12 == fourcc) {
        va_fourcc = VA_FOURCC_NV12;
    } else if (MFX_FOURCC_YV12 == fourcc) {
        va_fourcc = VA_FOURCC_YV12;
    } else if (MFX_FOURCC_P8 == fourcc) {
        va_fourcc = VA_FOURCC_P208;
    } else if (MFX_FOURCC_RGB4 == fourcc) {
        va_fourcc = VA_FOURCC_ARGB;
    }

    return va_fourcc;
}

mfxStatus SurfaceAllocator::convertToMfxStatus(VAStatus vaSts) {
    mfxStatus mfx_sts = MFX_ERR_NONE;

    switch (vaSts)
    {
    case VA_STATUS_ERROR_INVALID_DISPLAY:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_CONFIG:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_CONTEXT:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_SURFACE:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_BUFFER:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_IMAGE:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_INVALID_SUBPICTURE:
        mfx_sts = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_ALLOCATION_FAILED:
        mfx_sts = MFX_ERR_MEMORY_ALLOC;
        break;
    case VA_STATUS_ERROR_INVALID_PARAMETER:
        mfx_sts = MFX_ERR_INVALID_VIDEO_PARAM;
        break;
    case VA_STATUS_ERROR_ATTR_NOT_SUPPORTED:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_UNSUPPORTED_PROFILE:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_FLAG_NOT_SUPPORTED:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED:
        mfx_sts = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_SUCCESS:
        mfx_sts = MFX_ERR_NONE;
        break;
    default:
        mfx_sts = MFX_ERR_UNKNOWN;
        break;
    }
    return mfx_sts;
}

mfxStatus SurfaceAllocator::checkRequestType(mfxFrameAllocRequest *allocRequest)
{
    if (0 == allocRequest)
        return MFX_ERR_NULL_PTR;

    if ((allocRequest->Type & MFX_MEMTYPE_MASK) != 0) {
        if ((allocRequest->Type & (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET)) != 0) {
            return MFX_ERR_NONE;
        } else {
            return MFX_ERR_UNSUPPORTED;
        }
    } else {
        return MFX_ERR_UNSUPPORTED;
    }
}

mfxStatus SurfaceAllocator::releaseResponse(mfxFrameAllocResponse *allocResponse)
{
    vaapiContext *vaapiCtxs = NULL;
    VASurfaceID* surfacesPool = NULL;
    bool isP8Fourcc = false;

    if (!allocResponse) {
        return MFX_ERR_NULL_PTR;
    }

    if (allocResponse->mids) {
        vaapiCtxs = (vaapiContext*)(allocResponse->mids[0]);
        isP8Fourcc = (MFX_FOURCC_P8 == vaapiCtxs->m_va_fourcc) ? true : false;
        surfacesPool = vaapiCtxs->m_va_surface_id;
        for (mfxU32 i = 0; i < allocResponse->NumFrameActual; ++i) {
            if (MFX_FOURCC_P8 == vaapiCtxs[i].m_va_fourcc) {
                vaDestroyBuffer(m_va_display, surfacesPool[i]);
            }
        }
        free(vaapiCtxs);
        free(allocResponse->mids);
        allocResponse->mids = NULL;

        if (!isP8Fourcc) {
            vaDestroySurfaces(m_va_display, surfacesPool, allocResponse->NumFrameActual);
        }
        free(surfacesPool);
    }
    allocResponse->NumFrameActual = 0;
    return MFX_ERR_NONE;
}


mfxStatus SurfaceAllocator::allocImpl(mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse)
{
    mfxStatus mfx_sts = MFX_ERR_NONE;
    VAStatus  va_sts  = VA_STATUS_SUCCESS;
    unsigned int va_fourcc = 0;
    VASurfaceID* surfacesPool = NULL;
    vaapiContext *vaapiCtxs = NULL, *vaapiCtx = NULL;
    mfxMemId* mfx_mids = NULL;
    mfxU32 mfx_fourcc = allocRequest->Info.FourCC;
    mfxU16 surfaces_num = allocRequest->NumFrameSuggested, numAllocated = 0, i = 0;
    bool isSucceeded = false;

    memset(allocResponse, 0, sizeof(mfxFrameAllocResponse));

    va_fourcc = convertToVAFormat(mfx_fourcc);
    if (!va_fourcc || ((VA_FOURCC_YV12 != va_fourcc) && (VA_FOURCC_NV12 != va_fourcc) &&
                       (VA_FOURCC_ARGB != va_fourcc) &&(VA_FOURCC_YUY2 != va_fourcc) &&
                       (VA_FOURCC_P208 != va_fourcc))) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    if (!surfaces_num) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    surfacesPool = (VASurfaceID*)calloc(surfaces_num, sizeof(VASurfaceID));
    vaapiCtxs = (vaapiContext*)calloc(surfaces_num, sizeof(vaapiContext));
    mfx_mids = (mfxMemId*)calloc(surfaces_num, sizeof(mfxMemId));
    if ((NULL == surfacesPool) || (NULL == vaapiCtxs) || (NULL == mfx_mids)) {
        mfx_sts = MFX_ERR_MEMORY_ALLOC;
    }
    if (MFX_ERR_NONE == mfx_sts) {
        if( VA_FOURCC_P208 != va_fourcc ) {
            unsigned int format;
            VASurfaceAttrib va_attribute;
            memset(&va_attribute, 0, sizeof(VASurfaceAttrib));
            va_attribute.type = VASurfaceAttribPixelFormat;
            va_attribute.flags = VA_SURFACE_ATTRIB_SETTABLE;
            va_attribute.value.type = VAGenericValueTypeInteger;
            va_attribute.value.value.i = va_fourcc;
            format = va_fourcc;

            va_sts = vaCreateSurfaces(m_va_display, format, allocRequest->Info.Width, allocRequest->Info.Height,
                    surfacesPool, surfaces_num, &va_attribute, 1);

            mfx_sts = convertToMfxStatus(va_sts);
            isSucceeded = (MFX_ERR_NONE == mfx_sts);
        } else {
            VAContextID context_id = allocRequest->reserved[0];
            int width = SIZE_ALIGN32(allocRequest->Info.Width);
            int height = SIZE_ALIGN32(allocRequest->Info.Height);

            VABufferType codedbufType = VAEncCodedBufferType;
            int codedbufSize = static_cast<int>((width * height) * 400LL / (16 * 16));

            for (numAllocated = 0; numAllocated < surfaces_num; numAllocated++) {
                VABufferID codedBuf;

                va_sts = vaCreateBuffer(m_va_display, context_id, codedbufType, codedbufSize,
                                      1, NULL, &codedBuf);
                mfx_sts = convertToMfxStatus(va_sts);
                if (MFX_ERR_NONE != mfx_sts)
                    break;
                surfacesPool[numAllocated] = codedBuf;
            }
        }

    }
    if (MFX_ERR_NONE == mfx_sts) {
        for (i = 0; i < surfaces_num; ++i) {
            vaapiCtx = &(vaapiCtxs[i]);
            vaapiCtx->m_va_fourcc = mfx_fourcc;
            vaapiCtx->m_va_surface_id = &(surfacesPool[i]);
            mfx_mids[i] = vaapiCtx;
        }
    }
    if (MFX_ERR_NONE == mfx_sts) {
        allocResponse->mids = mfx_mids;
        allocResponse->NumFrameActual = surfaces_num;
    } else {
        allocResponse->mids = NULL;
        allocResponse->NumFrameActual = 0;
        if (VA_FOURCC_P208 != va_fourcc) {
            if (isSucceeded) {
                vaDestroySurfaces(m_va_display, surfacesPool, surfaces_num);
            }
        } else {
            for (i = 0; i < numAllocated; i++) {
                vaDestroyBuffer(m_va_display, surfacesPool[i]);
            }
        }
        if (mfx_mids) {
            free(mfx_mids);
            mfx_mids = NULL;
        }
        if (vaapiCtxs) {
            free(vaapiCtxs); vaapiCtxs = NULL;
        }
        if (surfacesPool) {
            free(surfacesPool); surfacesPool = NULL;
        }
    }
    return mfx_sts;
}

void SurfaceAllocator::mapNV12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image) {
    frameBuff->Pitch = (mfxU16)image->pitches[0];
    frameBuff->Y = mapBuff + image->offsets[0];
    frameBuff->U = mapBuff + image->offsets[1];
    frameBuff->V = frameBuff->U + 1;
}

void SurfaceAllocator::mapYV12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image) {
    frameBuff->Pitch = (mfxU16)image->pitches[0];
    frameBuff->Y = mapBuff + image->offsets[0];
    frameBuff->V = mapBuff + image->offsets[1];
    frameBuff->U = mapBuff + image->offsets[2];
}

void SurfaceAllocator::mapYUY12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image) {
    frameBuff->Pitch = (mfxU16)image->pitches[0];
    frameBuff->Y = mapBuff + image->offsets[0];
    frameBuff->U = frameBuff->Y + 1;
    frameBuff->V = frameBuff->Y + 3;
}

void SurfaceAllocator::mapRGB4Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image) {
    frameBuff->Pitch = (mfxU16)image->pitches[0];
    frameBuff->B = mapBuff + image->offsets[0];
    frameBuff->G = frameBuff->B + 1;
    frameBuff->R = frameBuff->B + 2;
    frameBuff->A = frameBuff->B + 3;
}

}
}
#endif
