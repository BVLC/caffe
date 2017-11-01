
#ifndef INCLUDE_CAFFE_UTIL_MSS_UTIL_SURFACE_ALLOCATOR_HPP_
#define INCLUDE_CAFFE_UTIL_MSS_UTIL_SURFACE_ALLOCATOR_HPP_

#ifdef USE_FFMPEG_QSV

#include <list>
#include <string.h>
#include <functional>
#include <va/va.h>
#include "mfxvideo.h"
#include <stdio.h>

namespace caffe { namespace mss {

enum {
    MFX_FOURCC_VP8_NV12    = MFX_MAKEFOURCC('V','P','8','N'),
    MFX_FOURCC_VP8_MBDATA  = MFX_MAKEFOURCC('V','P','8','M'),
    MFX_FOURCC_VP8_SEGMAP  = MFX_MAKEFOURCC('V','P','8','S'),
};

struct vaapiMemId
{
    VASurfaceID* m_surface;
    VAImage      m_image;
    unsigned int m_fourcc;
    mfxU8*       m_sys_buffer;
    mfxU8*       m_va_buffer;
};

class SurfaceAllocator : public mfxFrameAllocator
{
public:
	SurfaceAllocator();
    virtual ~SurfaceAllocator();

    mfxStatus init(VADisplay *dpy);
    mfxStatus close();

    virtual mfxStatus vaapiAllocFrames(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    virtual mfxStatus vaapiLockFrame(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus vaapiUnlockFrame(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus vaapiGetFrameHDL(mfxMemId mid, mfxHDL *handle);
    virtual mfxStatus vaapiFreeFrames(mfxFrameAllocResponse *response);

private:
    static mfxStatus MFX_CDECL  mfxAllocImpl(mfxHDL pthis, mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    static mfxStatus MFX_CDECL  mfxLockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL  mfxUnlockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL  mfxGetFrameHDLImpl(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL  mfxFreeImpl(mfxHDL pthis, mfxFrameAllocResponse *response);

    unsigned int convertToMfxFourcc(mfxU32 fourcc);
    mfxStatus convertToMfxStatus(VAStatus vaSts);
    unsigned int convertToVAFormat(mfxU32 fourcc);
    mfxStatus checkRequestType(mfxFrameAllocRequest *request);
    mfxStatus releaseResponse(mfxFrameAllocResponse *response);
    mfxStatus allocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
private:
    std::list<mfxFrameAllocResponse> m_responses;
    VADisplay m_dpy;
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV
#endif // INCLUDE_CAFFE_UTIL_MSS_UTIL_SURFACE_ALLOCATOR_HPP_
