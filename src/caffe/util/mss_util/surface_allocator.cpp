
#ifdef USE_FFMPEG_QSV

#include <assert.h>
#include <algorithm>
#include "caffe/util/mss_util/surface_allocator.hpp"

namespace caffe { namespace mss {

static const mfxU32 MFX_MEMTYPE_MASK = MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN | MFX_MEMTYPE_FROM_VPPOUT;

SurfaceAllocator::SurfaceAllocator() {
	m_dpy = NULL;
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
     m_dpy = *dpy;

    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::close() {
    std::list<mfxFrameAllocResponse> ::iterator itr;
    for (itr = m_responses.begin(); itr!= m_responses.end(); itr++)
    {
        releaseResponse(&*itr);
    }
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiAllocFrames(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    mfxStatus sts = MFX_ERR_NONE;
	if (0 == request || 0 == response || 0 == request->NumFrameSuggested)
        return MFX_ERR_MEMORY_ALLOC;

    if (MFX_ERR_NONE != checkRequestType(request))
        return MFX_ERR_UNSUPPORTED;
    m_responses.push_back(mfxFrameAllocResponse());

    sts = allocImpl(request, response);
    if (sts == MFX_ERR_NONE) {
        m_responses.back() = *response;
    } else {
        m_responses.pop_back();
    }
    return sts;
}

mfxStatus SurfaceAllocator::vaapiLockFrame(mfxMemId mid, mfxFrameData *ptr) {
    mfxStatus ret = MFX_ERR_NONE;
    VAStatus  vaSts  = VA_STATUS_SUCCESS;
    vaapiMemId* vaapiMid = (vaapiMemId*)mid;
    mfxU8* pBuffer = 0;

    if (!vaapiMid || !(vaapiMid->m_surface)) {
    	return MFX_ERR_INVALID_HANDLE;
    }

    mfxU32 mfx_fourcc = convertToMfxFourcc(vaapiMid->m_fourcc);

    if (MFX_FOURCC_P8 == mfx_fourcc) {
        VACodedBufferSegment *codedBuf;
        if (vaapiMid->m_fourcc == MFX_FOURCC_VP8_SEGMAP) {
        	vaSts = vaMapBuffer(m_dpy, *(vaapiMid->m_surface), (void **)(&pBuffer));
        } else {
        	vaSts = vaMapBuffer(m_dpy, *(vaapiMid->m_surface), (void **)(&codedBuf));
        }
        ret = convertToMfxStatus(vaSts);
        if (MFX_ERR_NONE == ret) {
            if (vaapiMid->m_fourcc == MFX_FOURCC_VP8_SEGMAP) {
                ptr->Y = pBuffer;
            } else {
                ptr->Y = (mfxU8*)codedBuf->buf;
            }
        }
    } else {
    	vaSts = vaSyncSurface(m_dpy, *(vaapiMid->m_surface));
        ret = convertToMfxStatus(vaSts);
        if (MFX_ERR_NONE == ret) {
        	vaSts = vaDeriveImage(m_dpy, *(vaapiMid->m_surface), &(vaapiMid->m_image));
        	ret = convertToMfxStatus(vaSts);
        }
        if (MFX_ERR_NONE == ret) {
        	vaSts = vaMapBuffer(m_dpy, vaapiMid->m_image.buf, (void **) &pBuffer);
            ret = convertToMfxStatus(vaSts);
        }
        if (MFX_ERR_NONE == ret) {
            switch (vaapiMid->m_image.format.fourcc) {
            case VA_FOURCC_NV12:
                if (mfx_fourcc == MFX_FOURCC_NV12) {
                    ptr->Pitch = (mfxU16)vaapiMid->m_image.pitches[0];
                    ptr->Y = pBuffer + vaapiMid->m_image.offsets[0];
                    ptr->U = pBuffer + vaapiMid->m_image.offsets[1];
                    ptr->V = ptr->U + 1;
                } else {
                	ret = MFX_ERR_LOCK_MEMORY;
                }
                break;
            case VA_FOURCC_YV12:
                if (mfx_fourcc == MFX_FOURCC_YV12) {
                    ptr->Pitch = (mfxU16)vaapiMid->m_image.pitches[0];
                    ptr->Y = pBuffer + vaapiMid->m_image.offsets[0];
                    ptr->V = pBuffer + vaapiMid->m_image.offsets[1];
                    ptr->U = pBuffer + vaapiMid->m_image.offsets[2];
                } else {
                	ret = MFX_ERR_LOCK_MEMORY;
                }
                break;
            case VA_FOURCC_YUY2:
                if (mfx_fourcc == MFX_FOURCC_YUY2) {
                    ptr->Pitch = (mfxU16)vaapiMid->m_image.pitches[0];
                    ptr->Y = pBuffer + vaapiMid->m_image.offsets[0];
                    ptr->U = ptr->Y + 1;
                    ptr->V = ptr->Y + 3;
                } else {
                	ret = MFX_ERR_LOCK_MEMORY;
                }
                break;
            case VA_FOURCC_ARGB:
                if (mfx_fourcc == MFX_FOURCC_RGB4) {
                    ptr->Pitch = (mfxU16)vaapiMid->m_image.pitches[0];
                    ptr->B = pBuffer + vaapiMid->m_image.offsets[0];
                    ptr->G = ptr->B + 1;
                    ptr->R = ptr->B + 2;
                    ptr->A = ptr->B + 3;
                } else {
                	ret = MFX_ERR_LOCK_MEMORY;
                }
                break;
        case VA_FOURCC_P208:
                if (mfx_fourcc == MFX_FOURCC_NV12) {
                    ptr->Pitch = (mfxU16)vaapiMid->m_image.pitches[0];
                    ptr->Y = pBuffer + vaapiMid->m_image.offsets[0];
                } else {
                	ret = MFX_ERR_LOCK_MEMORY;
                }
                break;
            default:
                ret = MFX_ERR_LOCK_MEMORY;
                break;
            }
        }
    }
    return ret;
}

mfxStatus SurfaceAllocator::vaapiUnlockFrame(mfxMemId mid, mfxFrameData *ptr) {
    vaapiMemId* vaapiMid = (vaapiMemId*)mid;

    if (!vaapiMid || !(vaapiMid->m_surface)) {
    	return MFX_ERR_INVALID_HANDLE;
    }

    mfxU32 mfx_fourcc = convertToMfxFourcc(vaapiMid->m_fourcc);

    if (MFX_FOURCC_P8 == mfx_fourcc) {
        vaUnmapBuffer(m_dpy, *(vaapiMid->m_surface));
    } else {
        vaUnmapBuffer(m_dpy, vaapiMid->m_image.buf);
        vaDestroyImage(m_dpy, vaapiMid->m_image.image_id);

        if (NULL != ptr) {
            ptr->Pitch = 0;
            ptr->Y     = NULL;
            ptr->U     = NULL;
            ptr->V     = NULL;
            ptr->A     = NULL;
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiGetFrameHDL(mfxMemId mid, mfxHDL *handle) {
    vaapiMemId* vaapiMid = (vaapiMemId*)mid;

    if (!handle || !vaapiMid || !(vaapiMid->m_surface)) {
    	return MFX_ERR_INVALID_HANDLE;
    }

    *handle = vaapiMid->m_surface;
    return MFX_ERR_NONE;
}

mfxStatus SurfaceAllocator::vaapiFreeFrames(mfxFrameAllocResponse *response) {
    if (response == 0)
        return MFX_ERR_INVALID_HANDLE;

    mfxStatus sts = MFX_ERR_NONE;

    std::list<mfxFrameAllocResponse>::iterator itr;
    for (itr = m_responses.begin(); itr != m_responses.end(); itr++) {
    	if ((*itr).mids != 0 && response->mids != 0 && (*itr).mids[0] == response->mids[0] &&
    			(*itr).NumFrameActual == response->NumFrameActual ) {
            sts = releaseResponse(response);
            m_responses.erase(itr);
            return sts;
    	}
    }

    // not found anywhere, report an error
    return MFX_ERR_INVALID_HANDLE;
}

mfxStatus SurfaceAllocator::mfxAllocImpl(mfxHDL pthis, mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator& self = *(SurfaceAllocator *)pthis;

    return self.vaapiAllocFrames(request, response);
}

mfxStatus SurfaceAllocator::mfxLockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator& self = *(SurfaceAllocator *)pthis;

    return self.vaapiLockFrame(mid, ptr);
}

mfxStatus SurfaceAllocator::mfxUnlockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator& self = *(SurfaceAllocator *)pthis;

    return self.vaapiUnlockFrame(mid, ptr);
}

mfxStatus SurfaceAllocator::mfxGetFrameHDLImpl(mfxHDL pthis, mfxMemId mid, mfxHDL *handle) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator& self = *(SurfaceAllocator *)pthis;

    return self.vaapiGetFrameHDL(mid, handle);
}

mfxStatus SurfaceAllocator::mfxFreeImpl(mfxHDL pthis, mfxFrameAllocResponse *response) {
    if (0 == pthis)
        return MFX_ERR_MEMORY_ALLOC;

    SurfaceAllocator& self = *(SurfaceAllocator *)pthis;

    return self.vaapiFreeFrames(response);
}

unsigned int SurfaceAllocator::convertToMfxFourcc(mfxU32 fourcc)
{
    switch (fourcc)
    {
    case MFX_FOURCC_VP8_NV12:
    case MFX_FOURCC_VP8_MBDATA:
        return MFX_FOURCC_NV12;
    case MFX_FOURCC_VP8_SEGMAP:
        return MFX_FOURCC_P8;

    default:
        return fourcc;
    }
}

unsigned int SurfaceAllocator::convertToVAFormat(mfxU32 fourcc)
{
    switch (fourcc)
    {
    case MFX_FOURCC_NV12:
        return VA_FOURCC_NV12;
    case MFX_FOURCC_YUY2:
        return VA_FOURCC_YUY2;
    case MFX_FOURCC_YV12:
        return VA_FOURCC_YV12;
    case MFX_FOURCC_RGB4:
        return VA_FOURCC_ARGB;
    case MFX_FOURCC_P8:
        return VA_FOURCC_P208;
    default:
        return 0;
    }
}

mfxStatus SurfaceAllocator::convertToMfxStatus(VAStatus vaSts) {
    mfxStatus ret = MFX_ERR_NONE;

    switch (vaSts)
    {
    case VA_STATUS_ERROR_INVALID_DISPLAY:
    case VA_STATUS_ERROR_INVALID_CONFIG:
    case VA_STATUS_ERROR_INVALID_CONTEXT:
    case VA_STATUS_ERROR_INVALID_SURFACE:
    case VA_STATUS_ERROR_INVALID_BUFFER:
    case VA_STATUS_ERROR_INVALID_IMAGE:
    case VA_STATUS_ERROR_INVALID_SUBPICTURE:
    	ret = MFX_ERR_NOT_INITIALIZED;
        break;
    case VA_STATUS_ERROR_ATTR_NOT_SUPPORTED:
    case VA_STATUS_ERROR_UNSUPPORTED_PROFILE:
    case VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT:
    case VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT:
    case VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE:
    case VA_STATUS_ERROR_FLAG_NOT_SUPPORTED:
    case VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED:
    	ret = MFX_ERR_UNSUPPORTED;
        break;
    case VA_STATUS_SUCCESS:
    	ret = MFX_ERR_NONE;
        break;
    case VA_STATUS_ERROR_ALLOCATION_FAILED:
    	ret = MFX_ERR_MEMORY_ALLOC;
        break;
    case VA_STATUS_ERROR_INVALID_PARAMETER:
    	ret = MFX_ERR_INVALID_VIDEO_PARAM;
    default:
    	ret = MFX_ERR_UNKNOWN;
        break;
    }
    return ret;
}

mfxStatus SurfaceAllocator::checkRequestType(mfxFrameAllocRequest *request)
{
    if (0 == request)
        return MFX_ERR_NULL_PTR;

    // check that MSS component is specified in request
    if ((request->Type & MFX_MEMTYPE_MASK) != 0) {
        if ((request->Type & (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET)) != 0) {
            return MFX_ERR_NONE;
        } else {
            return MFX_ERR_UNSUPPORTED;
        }
    } else {
        return MFX_ERR_UNSUPPORTED;
    }
}

mfxStatus SurfaceAllocator::releaseResponse(mfxFrameAllocResponse *response)
{
    vaapiMemId *vaapi_mids = NULL;
    VASurfaceID* surfaces = NULL;
    mfxU32 i = 0;
    bool isBitstreamMemory=false;

    if (!response) {
    	return MFX_ERR_NULL_PTR;
    }

    if (response->mids) {
        vaapi_mids = (vaapiMemId*)(response->mids[0]);
        mfxU32 mfx_fourcc = convertToMfxFourcc(vaapi_mids->m_fourcc);
        isBitstreamMemory = (MFX_FOURCC_P8 == mfx_fourcc)?true:false;
        surfaces = vaapi_mids->m_surface;
        for (i = 0; i < response->NumFrameActual; ++i) {
            if (MFX_FOURCC_P8 == vaapi_mids[i].m_fourcc) {
            	vaDestroyBuffer(m_dpy, surfaces[i]);
            } else if (vaapi_mids[i].m_sys_buffer) {
            	free(vaapi_mids[i].m_sys_buffer);
            }
        }
        free(vaapi_mids);
        free(response->mids);
        response->mids = NULL;

        if (!isBitstreamMemory) {
        	vaDestroySurfaces(m_dpy, surfaces, response->NumFrameActual);
        }
        free(surfaces);
    }
    response->NumFrameActual = 0;
    return MFX_ERR_NONE;
}


mfxStatus SurfaceAllocator::allocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response)
{
    mfxStatus mfx_res = MFX_ERR_NONE;
    VAStatus  va_res  = VA_STATUS_SUCCESS;
    unsigned int va_fourcc = 0;
    VASurfaceID* surfaces = NULL;
    VASurfaceAttrib attrib;
    vaapiMemId *vaapi_mids = NULL, *vaapi_mid = NULL;
    mfxMemId* mids = NULL;
    mfxU32 fourcc = request->Info.FourCC;
    mfxU16 surfaces_num = request->NumFrameSuggested, numAllocated = 0, i = 0;
    bool bCreateSrfSucceeded = false;

    memset(response, 0, sizeof(mfxFrameAllocResponse));
    memset(&attrib, 0, sizeof(VASurfaceAttrib));

    // VP8 hybrid driver has weird requirements for allocation of surfaces/buffers for VP8 encoding
    // to comply with them additional logic is required to support regular and VP8 hybrid allocation pathes
    mfxU32 mfx_fourcc = convertToMfxFourcc(fourcc);
    va_fourcc = convertToVAFormat(mfx_fourcc);
    if (!va_fourcc || ((VA_FOURCC_NV12 != va_fourcc) &&
                       (VA_FOURCC_YV12 != va_fourcc) &&
                       (VA_FOURCC_YUY2 != va_fourcc) &&
                       (VA_FOURCC_ARGB != va_fourcc) &&
                       (VA_FOURCC_P208 != va_fourcc))) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    if (!surfaces_num) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    if (MFX_ERR_NONE == mfx_res) {
        surfaces = (VASurfaceID*)calloc(surfaces_num, sizeof(VASurfaceID));
        vaapi_mids = (vaapiMemId*)calloc(surfaces_num, sizeof(vaapiMemId));
        mids = (mfxMemId*)calloc(surfaces_num, sizeof(mfxMemId));
        if ((NULL == surfaces) || (NULL == vaapi_mids) || (NULL == mids)) mfx_res = MFX_ERR_MEMORY_ALLOC;
    }
    if (MFX_ERR_NONE == mfx_res) {
        if( VA_FOURCC_P208 != va_fourcc ) {
            unsigned int format;

            attrib.type          = VASurfaceAttribPixelFormat;
            attrib.flags         = VA_SURFACE_ATTRIB_SETTABLE;
            attrib.value.type    = VAGenericValueTypeInteger;
            attrib.value.value.i = va_fourcc;
            format               = va_fourcc;

            if (fourcc == MFX_FOURCC_VP8_NV12) {
                attrib.type          = (VASurfaceAttribType)VASurfaceAttribUsageHint;
                attrib.value.value.i = VA_SURFACE_ATTRIB_USAGE_HINT_ENCODER;
            } else if (fourcc == MFX_FOURCC_VP8_MBDATA) {
                attrib.value.value.i = VA_FOURCC_P208;
                format               = VA_FOURCC_P208;
            } else if (va_fourcc == VA_FOURCC_NV12) {
                format = VA_RT_FORMAT_YUV420;
            }

            va_res = vaCreateSurfaces(m_dpy,
                                    format,
                                    request->Info.Width, request->Info.Height,
                                    surfaces,
                                    surfaces_num,
                                    &attrib, 1);

            mfx_res = convertToMfxStatus(va_res);
            bCreateSrfSucceeded = (MFX_ERR_NONE == mfx_res);
        } else {
            VAContextID context_id = request->reserved[0];
            int codedbuf_size;
            int width32 = 32 * ((request->Info.Width + 31) >> 5);
            int height32 = 32 * ((request->Info.Height + 31) >> 5);

            VABufferType codedbuf_type;
            if (fourcc == MFX_FOURCC_VP8_SEGMAP) {
                codedbuf_size = request->Info.Width * request->Info.Height;
                codedbuf_type = (VABufferType)VAEncMacroblockMapBufferType;
            } else {
                codedbuf_size = static_cast<int>((width32 * height32) * 400LL / (16 * 16));
                codedbuf_type = VAEncCodedBufferType;
            }

            for (numAllocated = 0; numAllocated < surfaces_num; numAllocated++) {
                VABufferID coded_buf;

                va_res = vaCreateBuffer(m_dpy,
                                      context_id,
                                      codedbuf_type,
                                      codedbuf_size,
                                      1,
                                      NULL,
                                      &coded_buf);
                mfx_res = convertToMfxStatus(va_res);
                if (MFX_ERR_NONE != mfx_res) break;
                surfaces[numAllocated] = coded_buf;
            }
        }

    }
    if (MFX_ERR_NONE == mfx_res) {
        for (i = 0; i < surfaces_num; ++i) {
            vaapi_mid = &(vaapi_mids[i]);
            vaapi_mid->m_fourcc = fourcc;
            vaapi_mid->m_surface = &(surfaces[i]);
            mids[i] = vaapi_mid;
        }
    }
    if (MFX_ERR_NONE == mfx_res) {
        response->mids = mids;
        response->NumFrameActual = surfaces_num;
    } else {
        response->mids = NULL;
        response->NumFrameActual = 0;
        if (VA_FOURCC_P208 != va_fourcc
            || fourcc == MFX_FOURCC_VP8_MBDATA ) {
            if (bCreateSrfSucceeded) vaDestroySurfaces(m_dpy, surfaces, surfaces_num);
        } else {
            for (i = 0; i < numAllocated; i++) {
                vaDestroyBuffer(m_dpy, surfaces[i]);
            }
        }
        if (mids) {
            free(mids);
            mids = NULL;
        }
        if (vaapi_mids) {
        	free(vaapi_mids); vaapi_mids = NULL;
        }
        if (surfaces) {
        	free(surfaces); surfaces = NULL;
        }
    }
    return mfx_res;
}


}
}
#endif
