
#ifdef USE_FFMPEG_QSV

#include <algorithm>
#include <limits>
#include <math.h>

#include "caffe/util/ffmpeg_qsv_utils.hpp"


namespace caffe { namespace qsv {

static AVBufferRef *g_device = NULL;

static mfxU16 msdk_atomic_add16(volatile mfxU16 *mem, mfxU16 val)
{
    asm volatile ("lock; xaddw %0,%1"
                  : "=r" (val), "=m" (*mem)
                  : "0" (val), "m" (*mem)
                  : "memory", "cc");
    return val;
}

mfxU16 msdk_atomic_inc16(volatile mfxU16 *pVariable)
{
    return msdk_atomic_add16(pVariable, 1) + 1;
}

/* Thread-safe 16-bit variable decrementing */
mfxU16 msdk_atomic_dec16(volatile mfxU16 *pVariable)
{
    return msdk_atomic_add16(pVariable, (mfxU16)-1) + 1;
}


static enum AVPixelFormat get_format(AVCodecContext *s,
                                     const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
    int ret;
    AVHWFramesContext *frames_ctx;
    AVQSVFramesContext *frames_hwctx;

    for (p = pix_fmts; *p != -1; p++) {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*p);
        if (!(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
            break;

        if (*p == AV_PIX_FMT_QSV) {
            av_buffer_unref(&s->hw_frames_ctx);
            s->hw_frames_ctx = av_hwframe_ctx_alloc(g_device);
            if (!s->hw_frames_ctx)
                return AV_PIX_FMT_NONE;

            frames_ctx   = (AVHWFramesContext*)s->hw_frames_ctx->data;
            frames_hwctx = (AVQSVFramesContext *)frames_ctx->hwctx;

            frames_ctx->width             = s->coded_width;
            frames_ctx->height            = s->coded_height;
            frames_ctx->format            = AV_PIX_FMT_QSV;
            frames_ctx->sw_format         = s->sw_pix_fmt;
            frames_ctx->initial_pool_size = 0;
            frames_hwctx->frame_type      = MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;

            ret = av_hwframe_ctx_init(s->hw_frames_ctx);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error initializing a QSV frame pool\n");
                return AV_PIX_FMT_NONE;
            }
            break;
        }
    }

    return *p;
}


QsvVideoReader::QsvVideoReader() {
    init();
}

QsvVideoReader::~QsvVideoReader() {
    
}

bool QsvVideoReader::open(const char* _filename) {
    unsigned i;
    bool valid = false;

    close();
    av_register_all();
    avfilter_register_all();
    avcodec_register_all();

    int err = av_hwdevice_ctx_create(&g_device, AV_HWDEVICE_TYPE_QSV,
                "/dev/dri/render128", NULL, 0);
    if (err < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to create QSV device.\n");
        goto exit_func;
    }

    av_dict_set(&dict, "rtsp_transport", "tcp", 0);
    err = avformat_open_input(&ic, _filename, NULL, &dict);
    if (err < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Error opening file %s\n", _filename);
        goto exit_func;
    }
    err = avformat_find_stream_info(ic, NULL);
    if (err < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Could not find codec parameters");
        goto exit_func;
    }
    for(i = 0; i < ic->nb_streams; i++)
    {
        if( AVMEDIA_TYPE_VIDEO == ic->streams[i]->codecpar->codec_type && video_stream < 0)
        {
            filter_ctx.dec_ctx = avcodec_alloc_context3(NULL);
            avcodec_parameters_to_context(filter_ctx.dec_ctx, ic->streams[i]->codecpar);
            
            filter_ctx.dec_ctx->framerate  = ic->streams[i]->avg_frame_rate;
            filter_ctx.dec_ctx->time_base  = av_inv_q(filter_ctx.dec_ctx->framerate);
            filter_ctx.dec_ctx->get_format = get_format;
            filter_ctx.dec_ctx->refcounted_frames = 1;
            
            AVCodec *codec = NULL;
            if (AV_CODEC_ID_H264 == filter_ctx.dec_ctx->codec_id) {
                codec = avcodec_find_decoder_by_name("h264_qsv");
                use_hw_decoder = true;
            } else if (AV_CODEC_ID_HEVC == filter_ctx.dec_ctx->codec_id) {
                codec = avcodec_find_decoder_by_name("hevc_qsv");
                use_hw_decoder = true;
            } else if (AV_CODEC_ID_VC1 == filter_ctx.dec_ctx->codec_id) {
                codec = avcodec_find_decoder_by_name("vc1_qsv");
                use_hw_decoder = true;
            } else if (AV_CODEC_ID_MPEG2VIDEO == filter_ctx.dec_ctx->codec_id) {
                codec = avcodec_find_decoder_by_name("mpeg2_qsv");
                use_hw_decoder = true;
            } else if (AV_CODEC_ID_VP8 == filter_ctx.dec_ctx->codec_id) {
                codec = avcodec_find_decoder_by_name("vp8_qsv");
                use_hw_decoder = true;
            } else {
                codec = avcodec_find_decoder(filter_ctx.dec_ctx->codec_id);
            }
            
            if (!codec || avcodec_open2(filter_ctx.dec_ctx, codec, NULL) < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error opening decoder failed \n");
                goto exit_func;
            }

            video_stream = i;
            break;
        }
    }
    if(video_stream >= 0) valid = true;

exit_func:
    if( !valid )
        close();

    return valid;
}

void QsvVideoReader::close() {
    if( img_convert_ctx )
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }
    av_frame_unref(&rgb_picture);
    if( filter_ctx.dec_ctx )
    {
        avcodec_close( filter_ctx.dec_ctx );
        avcodec_free_context(&(filter_ctx.dec_ctx));
    }
    if (filter_ctx.filter_graph)
        avfilter_graph_free(&(filter_ctx.filter_graph));

    if( ic )
    {
        avformat_close_input(&ic);
        ic = NULL;
    }

    // free last packet if exist
    if (packet.data) {
        av_packet_unref (&packet);
        packet.data = NULL;
    }

    if (dict != NULL)
       av_dict_free(&dict);

    init();
}

bool QsvVideoReader::isOpened() {
    return (video_stream >= 0);
}

void QsvVideoReader::seek(int64_t _frame_number) {
    _frame_number = std::min(_frame_number, get_total_frames());
    int delta = 16;

    // if we have not grabbed a single frame before first seek, let's read the first frame
    // and get some valuable information during the process
    if( first_frame_number < 0 && get_total_frames() > 1 )
        decodeFrame();

    for(;;)
    {
        int64_t _frame_number_temp = std::max(_frame_number-delta, (int64_t)0);
        double sec = (double)_frame_number_temp / get_fps();
        int64_t time_stamp = ic->streams[video_stream]->start_time;
        double  time_base  = r2d(ic->streams[video_stream]->time_base);
        time_stamp += (int64_t)(sec / time_base + 0.5);
        if (get_total_frames() > 1) av_seek_frame(ic, video_stream, time_stamp, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(filter_ctx.dec_ctx);
        if( _frame_number > 0 )
        {
            decodeFrame();

            if( _frame_number > 1 )
            {
                frame_number = dts_to_frame_number(picture_pts) - first_frame_number;

                if( frame_number < 0 || frame_number > _frame_number-1 )
                {
                    if( _frame_number_temp == 0 || delta >= INT_MAX/4 )
                        break;
                    delta = delta < 16 ? delta*2 : delta*3/2;
                    continue;
                }
                while( frame_number < _frame_number-1 )
                {
                    if(!decodeFrame())
                        break;
                }
                frame_number++;
                break;
            }
            else
            {
                frame_number = 1;
                break;
            }
        }
        else
        {
            frame_number = 0;
            break;
        }
    }
}

int64_t QsvVideoReader::get_total_frames() {
    int64_t nbf = ic->streams[video_stream]->nb_frames;

    if (nbf == 0)
    {
        nbf = (int64_t)floor(get_duration_sec() * get_fps() + 0.5);
    }
    return nbf;
}

void QsvVideoReader::getVideoFrame(cv::Mat &cv_img, cv::Mat &org_img) {
    decodeFrame();
    if (frame.data) {
        IplImage imgFrame;
        cvInitImageHeader(&imgFrame, cvSize(frame.width, frame.height), 8, frame.cn);
        cvSetData(&imgFrame, frame.data, frame.step);
        cv::cvarrToMat(&imgFrame).copyTo(cv_img);
    }
    if (org_frame.data) {
        IplImage imgFrame;
        cvInitImageHeader(&imgFrame, cvSize(org_frame.width, org_frame.height), 8, org_frame.cn);
        cvSetData(&imgFrame, org_frame.data, org_frame.step);
        cv::cvarrToMat(&imgFrame).copyTo(org_img);
    }
}

void QsvVideoReader::setResize(int width, int height, int layout) {
    out_width = width;
    out_height = height;
    out_layout = layout;
}

void QsvVideoReader::init() {
    ic = 0;
    video_stream = -1;
    picture_pts = ((int64_t)AV_NOPTS_VALUE);
    first_frame_number = -1;
    memset( &rgb_picture, 0, sizeof(rgb_picture) );
    memset( &frame, 0, sizeof(frame) );
    memset( &org_frame, 0, sizeof(org_frame) );
    filename = 0;
    memset(&packet, 0, sizeof(packet));
    av_init_packet(&packet);
    img_convert_ctx = 0;

    frame_number = 0;
    eps_zero = 0.000025;

    dict = NULL;
    memset(&filter_ctx, 0, sizeof(filter_ctx));
    out_width = 0;
    out_height = 0;
    out_layout = 1;
    use_hw_decoder = false;
}

double QsvVideoReader::get_duration_sec() {
    double sec = (double)ic->duration / (double)AV_TIME_BASE;

    if (sec < eps_zero)
    {
        sec = (double)ic->streams[video_stream]->duration * r2d(ic->streams[video_stream]->time_base);
    }

    if (sec < eps_zero)
    {
        sec = (double)ic->streams[video_stream]->duration * r2d(ic->streams[video_stream]->time_base);
    }

    return sec;
}

double  QsvVideoReader::r2d(AVRational r) {
    return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

double QsvVideoReader::get_fps()
{
    double fps = r2d(ic->streams[video_stream]->avg_frame_rate);

    if (fps < eps_zero)
    {
        fps = r2d(ic->streams[video_stream]->avg_frame_rate);
    }


    if (fps < eps_zero)
    {
        fps = 1.0 / r2d(filter_ctx.dec_ctx->time_base);
    }

    return fps;
}

bool QsvVideoReader::decodeFrame() {
    bool valid = false;
    int got_picture;

    int count_errs = 0;
    const int max_number_of_attempts = 1 << 9;

    if( !ic || !(filter_ctx.dec_ctx) )  return false;

    if( ic->streams[video_stream]->nb_frames > 0 &&
        frame_number > ic->streams[video_stream]->nb_frames )
        return false;
    picture_pts = ((int64_t)AV_NOPTS_VALUE);
    
    // get the next frame
    while (!valid)
    {

        av_packet_unref (&packet);

        int ret = av_read_frame(ic, &packet);
        if (ret == AVERROR(EAGAIN)) continue;

        if( packet.stream_index != video_stream )
        {
            av_packet_unref (&packet);
            count_errs++;
            if (count_errs > max_number_of_attempts)
                break;
            continue;
        }
        AVFrame *decode_frame = av_frame_alloc();
        // Decode video frame
        int64_t dec_start = av_gettime();
        avcodec_decode_video2(filter_ctx.dec_ctx, decode_frame, &got_picture, &packet);

        // Did we get a video frame?
        if(got_picture)
        {
            //picture_pts = picture->best_effort_timestamp;
            if( picture_pts == ((int64_t)AV_NOPTS_VALUE) )
                picture_pts = decode_frame->pkt_pts != ((int64_t)AV_NOPTS_VALUE) && decode_frame->pkt_pts != 0 ? decode_frame->pkt_pts : decode_frame->pkt_dts;

            frame_number++;
            valid = true;
#ifdef USE_VPP_QSV
            if (!filter_ctx.initiallized) {
                ret = init_filter(&filter_ctx, filter_ctx.dec_ctx, av_asprintf("vpp_qsv=w=%d:h=%d", filter_ctx.dec_ctx->width, filter_ctx.dec_ctx->height));
                if (ret < 0) {
                    av_frame_free(&decode_frame);
                    return false;
                }
            }
            av_log(NULL, AV_LOG_INFO, "Pushing decoded frame to filters\n");
            /* push the decoded frame into the filtergraph */
            ret = av_buffersrc_add_frame_flags(filter_ctx.buffersrc_ctx, decode_frame, 0);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error while feeding the filtergraph\n");
                av_frame_free(&decode_frame);
                return false;
            }
            av_log(NULL, AV_LOG_INFO, "Pulling filtered frame from filters\n");
            AVFrame *small_frame = av_frame_alloc();
            ret = av_buffersink_get_frame(filter_ctx.buffersink_ctx,
                    small_frame);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Pulling filtered frame from filters has error\n");
                /* if no more frames for output - returns AVERROR(EAGAIN)
                 * if flushed and no more frames for output - returns AVERROR_EOF
                 * rewrite retcode to 0 to show it as normal procedure completion
                 */
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    ret = 0;
                } else {
                    av_frame_free(&decode_frame);
                    av_frame_free(&small_frame);
                    return false;
                }
            }
            AVFrame *copy_frame = av_frame_alloc();
            copy_frame->format = AV_PIX_FMT_NONE;
            ret = av_hwframe_transfer_data(copy_frame, small_frame, 0);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Failed to transfer data to "
                       "output frame: %d.\n", ret);
                av_frame_free(&decode_frame);
                av_frame_free(&small_frame);
                av_frame_free(&copy_frame);
                return false;
            }
            ret = av_frame_copy_props(copy_frame, small_frame);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Failed to copy props to "
                       "output frame: %d.\n", ret);
                av_frame_free(&decode_frame);
                av_frame_free(&small_frame);
                av_frame_free(&copy_frame);
                return false;
            }
            av_frame_unref(small_frame);
            av_frame_move_ref(small_frame, copy_frame);
            if (small_frame->format != -1) {
                init_swscontext((enum AVPixelFormat)small_frame->format);
                sws_scale(
                    img_convert_ctx,
                    small_frame->data,
                    small_frame->linesize,
                    0, out_height == 0 ? filter_ctx.dec_ctx->height : out_height,
                    rgb_picture.data,
                    rgb_picture.linesize
                    );
            }
            av_frame_free(&copy_frame);
            av_frame_free(&small_frame);
#else
              if (decode_frame->data[3]) {
                  mfxFrameSurface1* surface = (mfxFrameSurface1*)decode_frame->data[3];
                if (!mss_pipeline.isInitialized()) {
                    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)g_device->data;
                    AVQSVDeviceContext *hwctx = (AVQSVDeviceContext *)device_ctx->hwctx;
                    mss_pipeline.initPipeline(decode_frame, out_width, out_height, out_layout, hwctx);

                    mfxFrameSurface1* surface = (mfxFrameSurface1*)decode_frame->data[3];
                    org_frame.step = surface->Info.CropW*3;
                    org_frame.cn = 3;
                    org_frame.width = surface->Info.CropW;
                    org_frame.height = surface->Info.CropH;
                }
                if (mss_pipeline.isInitialized()) {
                    AVFrame *copy_frame = av_frame_alloc();
                    copy_frame->format = AV_PIX_FMT_NONE;
                    ret = av_hwframe_transfer_data(copy_frame, decode_frame, 0);
                    if (ret < 0) {
                        av_log(NULL, AV_LOG_ERROR, "Failed to transfer data to "
                               "output frame: %d.\n", ret);
                        av_frame_free(&decode_frame);
                        av_frame_free(&copy_frame);
                        return false;
                    }
                    ret = av_frame_copy_props(copy_frame, decode_frame);
                    if (ret < 0) {
                        av_log(NULL, AV_LOG_ERROR, "Failed to copy props to "
                               "output frame: %d.\n", ret);
                        av_frame_free(&decode_frame);
                        av_frame_free(&copy_frame);
                        return false;
                    }
                    av_frame_unref(decode_frame);
                    av_frame_move_ref(decode_frame, copy_frame);
                    mss_pipeline.processFrame(decode_frame, &frame, &org_frame);
                    av_frame_free(&copy_frame);
                } else {
                    printf("MSS pipeline init failed!!\n");
                }
              }
#endif
        }
        else
        {
            count_errs++;
            if (count_errs > max_number_of_attempts) {
                av_frame_free(&decode_frame);
                break;
            }
        }
        av_frame_free(&decode_frame);
        int64_t dec_end = av_gettime();
        //printf("@@@@@@@@@ qsv decode and vpp cost [%ld] ms\n", (dec_end-dec_start));
    }

    if( valid && first_frame_number < 0 )
        first_frame_number = dts_to_frame_number(picture_pts);

    return valid;
}

double QsvVideoReader::dts_to_sec(int64_t dts) {
    return (double)(dts - ic->streams[video_stream]->start_time) *
    r2d(ic->streams[video_stream]->time_base);
}
int64_t QsvVideoReader::dts_to_frame_number(int64_t dts) {
    double sec = dts_to_sec(dts);
    return (int64_t)(get_fps() * sec + 0.5);
}

int QsvVideoReader::init_filter(FilteringContext* fctx, AVCodecContext *dec_ctx, const char *filter_spec)
{
    char args[512];
    int ret = 0;
    AVFilter *buffersrc = NULL;
    AVFilter *buffersink = NULL;
    AVFilterContext *buffersrc_ctx = NULL;
    AVFilterContext *buffersink_ctx = NULL;
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs  = avfilter_inout_alloc();
    AVFilterGraph *filter_graph = avfilter_graph_alloc();
    
    frame.width = dec_ctx->coded_width;
    frame.height = dec_ctx->coded_height;
    frame.cn = 3;
    frame.step = 0;
    frame.data = NULL;
    
    if (!outputs || !inputs || !filter_graph) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    if (dec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
        buffersrc = avfilter_get_by_name("buffer");
        buffersink = avfilter_get_by_name("buffersink");
        if (!buffersrc || !buffersink) {
            av_log(NULL, AV_LOG_ERROR, "filtering source or sink element not found\n");
            ret = AVERROR_UNKNOWN;
            goto end;
        }

        snprintf(args, sizeof(args),
                "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d"
                ":frame_rate=%d/%d",
                dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
                dec_ctx->time_base.num, dec_ctx->time_base.den,
                dec_ctx->sample_aspect_ratio.num,
                dec_ctx->sample_aspect_ratio.den,
                dec_ctx->framerate.num, dec_ctx->framerate.den);

        ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in",
                args, NULL, filter_graph);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot create buffer source\n");
            goto end;
        }

        if (dec_ctx->hw_frames_ctx) {
            AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();
            par->hw_frames_ctx = dec_ctx->hw_frames_ctx;
            ret = av_buffersrc_parameters_set(buffersrc_ctx, par);
            av_freep(&par);
            if (ret < 0)
                goto end;
        }

        ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                NULL, NULL, filter_graph);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot create buffer sink\n");
            goto end;
        }
    } else {
        ret = AVERROR_UNKNOWN;
        goto end;
    }

    /* Endpoints for the filter graph. */
    outputs->name       = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx    = 0;
    outputs->next       = NULL;

    inputs->name       = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx    = 0;
    inputs->next       = NULL;

    if (!outputs->name || !inputs->name) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    if ((ret = avfilter_graph_parse_ptr(filter_graph, filter_spec,
                    &inputs, &outputs, NULL)) < 0)
        goto end;

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0)
        goto end;

    /* Fill FilteringContext */
    fctx->buffersrc_ctx = buffersrc_ctx;
    fctx->buffersink_ctx = buffersink_ctx;
    fctx->filter_graph = filter_graph;
    fctx->initiallized = 1;

end:
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    return ret;
}

bool QsvVideoReader::init_swscontext(enum AVPixelFormat dec_pix_fmts) {
    if( img_convert_ctx == NULL ||
        frame.data == NULL )
    {
        // Some sws_scale optimizations have some assumptions about alignment of data/step/width/height
        // Also we use coded_width/height to workaround problem with legacy ffmpeg versions (like n0.8)
        int buffer_width = out_width == 0 ? filter_ctx.dec_ctx->width : out_width;
        int buffer_height = out_height == 0 ? filter_ctx.dec_ctx->height : out_height;

        img_convert_ctx = sws_getCachedContext(
                img_convert_ctx,
                buffer_width, buffer_height,
                dec_pix_fmts,
                buffer_width, buffer_height,
                AV_PIX_FMT_BGR24,
                SWS_BICUBIC,
                NULL, NULL, NULL
                );

        if (img_convert_ctx == NULL)
            return false;

        av_frame_unref(&rgb_picture);
        rgb_picture.format = AV_PIX_FMT_BGR24;
        rgb_picture.width = buffer_width;
        rgb_picture.height = buffer_height;
        if (0 != av_frame_get_buffer(&rgb_picture, 4))
        {
            return false;
        }
        frame.width = buffer_width;
        frame.height = buffer_height;
        frame.cn = 3;
        frame.data = rgb_picture.data[0];
        frame.step = rgb_picture.linesize[0];
    }
    return true;
}

}
}

#endif
