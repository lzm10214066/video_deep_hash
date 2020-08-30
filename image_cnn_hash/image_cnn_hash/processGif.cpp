extern "C"{

#ifdef __cplusplus
 #define __STDC_CONSTANT_MACROS
 #ifdef _STDINT_H
  #undef _STDINT_H
 #endif
 #include <stdint.h>
#endif

}

#include "processGif.h"
#include <stdio.h>
#include <stdlib.h>


ProGif::ProGif(int _max_num, double _th, int _timeStep) :max_num(_max_num), th(_th), timeStep(_timeStep)
{

}

int innerprocess_pickGif2Mats(
	const char*        filename, 
	int                max_num, 
	vector<Mat>        &gif_frames, 
        AVFormatContext*   pFormatCtx,
        AVFrame*           pFrame, 
	AVFrame*           pFrameYUV,
        AVPacket*          packet,
        struct SwsContext* img_convert_ctx,
	int timeStep)
{
	AVCodecContext  *pCodecCtx;
	AVCodec         *pCodec;

	int ret_type = 0;  // 0 for image2, 1 for gif
	int i, videoindex;
	int ret, got_picture;

	
	if (avformat_open_input(&pFormatCtx, filename, NULL, NULL) != 0)
	{
		printf("Couldn't open input stream.\n");
		return -2;
	}

	if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
	{
		printf("Couldn't find stream information.\n");
		return -1;
	}

	videoindex = -1;

	for (i = 0; i < pFormatCtx->nb_streams; i++)
	{
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoindex = i;
			break;
		}
	}

	if (videoindex == -1)
	{
		printf("Didn't find a video stream.\n");
		return -1;
	}

	pCodecCtx = pFormatCtx->streams[videoindex]->codec;
	pCodec = avcodec_find_decoder(pCodecCtx->codec_id);  // AV_CODEC_ID_GIF   

	if (pCodec == NULL)
	{
		avcodec_close(pCodecCtx);	
		printf("Codec not found.\n");
		return -1;
	}

	if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
	{
		avcodec_close(pCodecCtx);	
		printf("Could not open codec.\n");
		return -1;
	}


	int size = avpicture_get_size(AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);
	if(pCodecCtx->width == 0 || pCodecCtx->height == 0)
	{
		avcodec_close(pCodecCtx);	
		printf("Error loading images with ffmpeg!\n");
		return -1;
	}

	uint8_t *out_buffer = (uint8_t *)av_malloc(size * 2); // 申请空间
	avpicture_fill((AVPicture *)pFrameYUV, out_buffer, AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);

	//Output Info-----------------------------  
	//printf("--------------- File Information ----------------\n");
	//av_dump_format(pFormatCtx, 0, filename, 0);
	//cout << pFormatCtx->iformat->name << endl;
	//dump_stream_format(pFormatCtx, 0, 0, 0);
	//printf("-------------------------------------------------\n");
	ret_type = (string)pFormatCtx->iformat->name == "gif" ? 1 : 0;


        if(pCodecCtx->width == 0 || pCodecCtx->height == 0 || pCodecCtx->pix_fmt == -1)
        {
                printf("Error loading images with ffmpeg!\n");
                return -1;
        }	


	img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
		pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_BGR24,
		SWS_BICUBIC, NULL, NULL, NULL);


	
	ret = 1;

	int nFramePerMinite = 3;
	while (gif_frames.size() < max_num && ret >= 0)
	{
		int step = timeStep;
		ret = av_read_frame(pFormatCtx, packet);
		if (ret < 0)
		{
			break;
		}

		if (packet->stream_index == videoindex)
		{
			ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);

			if (ret < 0)
			{
				printf("Decode Error.\n");
				break;
			}

			//提取i帧
			if (got_picture && pFrame->pict_type == AV_PICTURE_TYPE_I)
			{
				try
				{
					sws_scale(img_convert_ctx, (const uint8_t* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
					pFrameYUV->data, pFrameYUV->linesize);

					Mat show(pFrame->height, pFrame->width, CV_8UC3);
					memcpy(show.data, out_buffer, size);
	
					// 保存视频关键帧，调试用
					/*stringstream sstream;
					sstream << gif_frames.size();
					string savePath = "output" + sstream.str() + ".jpeg";
					imwrite(savePath, show);*/

					gif_frames.push_back(show);
				}
				catch(...)
				{
				}
			}
		}
		av_free_packet(packet);

		while (step--)
		{
			ret = av_read_frame(pFormatCtx, packet);
			if (ret < 0)
			{
				break;
			}
			if (packet->stream_index == videoindex)
			{
				ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);

				if (ret < 0)
				{
					printf("Decode Error in while.\n");
					av_free_packet(packet);
					break;
				}
			}
			av_free_packet(packet);
		}


	}

	sws_freeContext(img_convert_ctx);
	avcodec_close(pCodecCtx);	
	av_free(out_buffer);
	return ret_type;

}


int ProGif::pickGif2Mats(const char* filename, int max_num, vector<Mat> &gif_frames)
{
	
	av_register_all();
	AVFormatContext *pFormatCtx=NULL;
	AVFrame *pFrame=NULL, *pFrameYUV=NULL;
	AVPacket packet;
	struct SwsContext *img_convert_ctx=NULL;


	pFormatCtx = avformat_alloc_context(); // 申请空间
	pFrame     = av_frame_alloc();
	pFrameYUV  = av_frame_alloc();

	int ret = -1;	
	ret = innerprocess_pickGif2Mats(filename, max_num, gif_frames, pFormatCtx, pFrame, pFrameYUV, &packet, img_convert_ctx, timeStep);
	av_frame_free(&pFrameYUV);	

	av_frame_free(&pFrame);
        if (ret != -2)
                avformat_close_input(&pFormatCtx);


	//avformat_free_context(pFormatCtx);
	return ret;
}

double ProGif::simiTwoImages(Mat &_imga, Mat &_imgb)
{
	Mat imga = _imga.clone();
	Mat imgb = _imgb.clone();
	if (imga.channels() != 1) cvtColor(imga, imga, CV_BGR2GRAY);
	if (imgb.channels() != 1) cvtColor(imgb, imgb, CV_BGR2GRAY);

	Mat diff;
	absdiff(imga, imgb, diff);
	threshold(diff, diff, 70, 255, THRESH_BINARY);
	//diff /= 8;
	int non_zero = countNonZero(diff);
	double simi = 1-double(non_zero) / (diff.rows*diff.cols);
	return simi;
}

void ProGif::duplicateFilter(vector<Mat> &frames, double th)
{
	vector<Mat> res;
	vector<bool> pass(frames.size(), false);
	for (int i = 0; i < frames.size(); ++i)
	{
		if (pass[i]) continue;
		for (int j = i+1; j < frames.size(); ++j)
		{
			if (pass[j]) continue;
			double simi = simiTwoImages(frames[i], frames[j]);
			if (simi > th) pass[j] = true;
		}
	}
	for (int i = 0; i < frames.size();++i)
	{
		if (!pass[i]) res.push_back(frames[i]);
	}
	frames.clear();
	frames = res;
}

int ProGif::proGif(string &fileName, vector<Mat> &frames)
{
	int ret = pickGif2Mats(fileName.c_str(), max_num, frames);
	//duplicateFilter(frames, th);
	return ret;
}
