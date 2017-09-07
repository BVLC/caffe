# - Find ffmpegQSV
#
#  ffmpegQSV_INCLUDES  - List of ffmpegQSV includes
#  ffmpegQSV_mfx_INCLUDES  - List of ffmpegQSV includes
#  ffmpegQSV_avcodec_LIBRARIES - List of libraries when using ffmpegQSV.
#  ffmpegQSV_avfilter_LIBRARIES - List of libraries when using ffmpegQSV.
#  ffmpegQSV_avformat_LIBRARIES - List of libraries when using ffmpegQSV.
#  ffmpegQSV_avutil_LIBRARIES - List of libraries when using ffmpegQSV.
#  ffmpegQSV_FOUND     - True if ffmpegQSV found.

# Look for the header file.
find_path(ffmpegQSV_INCLUDE NAMES libavcodec/qsv.h libavdevice/version.h libavfilter/version.h libavformat/version.h libavutil/hwcontext_qsv.h 
                            PATHS $ENV{FFMPEG_QSV_ROOT}/include /usr/local/include /usr/include 
                            DOC "Path in which the file qsv include files is located." )
find_path(ffmpegQSV_mfx_INCLUDE NAMES mfx/mfxvideo.h 
                            PATHS /opt/intel/mediasdk/include /usr/local/include /usr/include 
                            DOC "Path in which the file media SDK include files is located." )
# Look for the library.
find_library(ffmpegQSV_avcodec_LIBRARY NAMES avcodec
                                       PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 $ENV{FFMPEG_QSV_ROOT}/lib
                                       DOC "Path to ffmpegQSV library." )
#find_library(ffmpegQSV_avdevice_LIBRARY NAMES avdevice
#                                       PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 $ENV{FFMPEG_QSV_ROOT}/lib
#                                       DOC "Path to ffmpegQSV library." )
find_library(ffmpegQSV_avfilter_LIBRARY NAMES avfilter
                                       PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 $ENV{FFMPEG_QSV_ROOT}/lib
                                       DOC "Path to ffmpegQSV library." )
find_library(ffmpegQSV_avformat_LIBRARY NAMES avformat 
                                       PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 $ENV{FFMPEG_QSV_ROOT}/lib
                                       DOC "Path to ffmpegQSV library." )
find_library(ffmpegQSV_avutil_LIBRARY NAMES avutil
                                       PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 $ENV{FFMPEG_QSV_ROOT}/lib
                                       DOC "Path to ffmpegQSV library." )
find_library(ffmpegQSV_mfx_LIBRARY NAMES dispatch_shared
                                       PATHS /opt/intel/mediasdk/lib/lin_x64
                                       DOC "Path to MSS library." )
find_library(ffmpegQSV_libva_LIBRARY NAMES va
                                       PATHS /usr/lib /usr/lib64
                                       DOC "Path to libva library." )
find_library(ffmpegQSV_libva-drm_LIBRARY NAMES va-drm
                                       PATHS /usr/lib /usr/lib64
                                       DOC "Path to libva-drm library." )
find_library(ffmpegQSV_dl_LIBRARY NAMES dl
                                       PATHS /usr/lib /usr/lib64
                                       DOC "Path to libdl library." )
find_library(ffmpegQSV_rt_LIBRARY NAMES rt
                                       PATHS /usr/lib /usr/lib64
                                       DOC "Path to librt library." )

#set(ffmpegQSV_LIBRARY "${ffmpegQSV_avcodec_LIBRARY} ${ffmpegQSV_avdevice_LIBRARY} ${ffmpegQSV_avfilter_LIBRARY} ${ffmpegQSV_avformat_LIBRARY} ${ffmpegQSV_avutil_LIBRARY}")
set(ffmpegQSV_LIBRARY "${ffmpegQSV_avcodec_LIBRARY} ${ffmpegQSV_avfilter_LIBRARY} ${ffmpegQSV_avformat_LIBRARY} ${ffmpegQSV_avutil_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ffmpegQSV DEFAULT_MSG ffmpegQSV_INCLUDE ffmpegQSV_LIBRARY)

if(FFMPEGQSV_FOUND)
  message(STATUS "Found ffmpegQSV (include: ${ffmpegQSV_INCLUDE}, library: ${ffmpegQSV_LIBRARY})")
  set(ffmpegQSV_INCLUDES ${ffmpegQSV_INCLUDE})
  set(ffmpegQSV_mfx_INCLUDES ${ffmpegQSV_mfx_INCLUDE})
  set(ffmpegQSV_avcodec_LIBRARIES ${ffmpegQSV_avcodec_LIBRARY})
  set(ffmpegQSV_avfilter_LIBRARIES ${ffmpegQSV_avfilter_LIBRARY})
  set(ffmpegQSV_avformat_LIBRARIES ${ffmpegQSV_avformat_LIBRARY})
  set(ffmpegQSV_avutil_LIBRARIES ${ffmpegQSV_avutil_LIBRARY})
  set(ffmpegQSV_mfx_LIBRARIES ${ffmpegQSV_mfx_LIBRARY})
  set(ffmpegQSV_libva_LIBRARIES ${ffmpegQSV_libva_LIBRARY})
  set(ffmpegQSV_libva-drm_LIBRARIES ${ffmpegQSV_libva-drm_LIBRARY})
  set(ffmpegQSV_dl_LIBRARIES ${ffmpegQSV_dl_LIBRARY})
  set(ffmpegQSV_rt_LIBRARIES ${ffmpegQSV_rt_LIBRARY})
  
  mark_as_advanced(ffmpegQSV_INCLUDE ffmpegQSV_LIBRARY)
endif()
