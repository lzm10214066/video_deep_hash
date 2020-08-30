//
//  ImagePreProcess.h
//  ImageRetrievalEngine
//
//  Created by XuRui on 15-3-21.
//  Copyright (c) 2015Äê XuRui. All rights reserved.
//

#ifndef __ImageRetrievalEngine__ImagePreProcess__
#define __ImageRetrievalEngine__ImagePreProcess__

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

class ImagePreProcess {
public:
	void HSVimgQuantization(const cv::Mat& hsv_img, cv::Mat& quantized_img);
	void ImgResize(const cv::Mat& img, cv::Mat& resized_img, const int obj_len);

	void imgRGB_HSVQuantization(const Mat &img_rgb, std::vector<int> &code);
};
#endif /* defined(__ImageRetrievalEngine__ImagePreProcess__) */