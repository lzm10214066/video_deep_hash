
#ifndef _GIF_FFMPEG_
#define _GIF_FFMPEG_

#include <stdio.h>  

#define __STDC_CONSTANT_MACROS  

#ifdef _WIN64
//Windows  
extern "C"
{
#include "libavcodec/avcodec.h"  
#include "libavformat/avformat.h"  
#include "libswscale/swscale.h"  
};
#else  
//Linux...  
#ifdef __cplusplus  
extern "C"
{
#endif  
#include <libavcodec/avcodec.h>  
#include <libavformat/avformat.h>  
#include <libswscale/swscale.h>  
#ifdef __cplusplus  
};
#endif  
#endif  

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


//int gif2Mats(const string &filePath, vector<Mat> &frames);


#endif
