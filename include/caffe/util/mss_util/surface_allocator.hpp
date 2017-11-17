
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

struct vaapiContext
{
    VASurfaceID* m_va_surface_id;
    unsigned int m_va_fourcc;
    VAImage m_va_image;
};

class SurfaceAllocator : public mfxFrameAllocator
{
public:
    SurfaceAllocator();
    virtual ~SurfaceAllocator();

    mfxStatus init(VADisplay *dpy);
    mfxStatus close();

    virtual mfxStatus vaapiAllocFrames(mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse);
    virtual mfxStatus vaapiLockFrame(mfxMemId context, mfxFrameData *frameBuff);
    virtual mfxStatus vaapiUnlockFrame(mfxMemId context, mfxFrameData *frameBuff);
    virtual mfxStatus vaapiGetFrameHDL(mfxMemId context, mfxHDL *handle);
    virtual mfxStatus vaapiFreeFrames(mfxFrameAllocResponse *allocResponse);

private:
    static mfxStatus MFX_CDECL  mfxAllocImpl(mfxHDL pthis, mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse);
    static mfxStatus MFX_CDECL  mfxLockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL  mfxUnlockImpl(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL  mfxGetFrameHDLImpl(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL  mfxFreeImpl(mfxHDL pthis, mfxFrameAllocResponse *allocResponse);

    mfxStatus convertToMfxStatus(VAStatus vaSts);
    unsigned int convertToVAFormat(mfxU32 fourcc);
    mfxStatus checkRequestType(mfxFrameAllocRequest *allocRequest);
    mfxStatus releaseResponse(mfxFrameAllocResponse *allocResponse);
    mfxStatus allocImpl(mfxFrameAllocRequest *allocRequest, mfxFrameAllocResponse *allocResponse);

    void mapNV12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image);
    void mapYV12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image);
    void mapYUY12Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image);
    void mapRGB4Buffer(mfxFrameData *frameBuff, unsigned char* mapBuff, VAImage *image);
private:
    std::list<mfxFrameAllocResponse> m_surface_pool;
    VADisplay m_va_display;
};

} // namespace mss
} // namespace caffe

#endif // USE_FFMPEG_QSV
#endif // INCLUDE_CAFFE_UTIL_MSS_UTIL_SURFACE_ALLOCATOR_HPP_
