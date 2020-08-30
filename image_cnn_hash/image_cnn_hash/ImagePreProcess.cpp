//
//  ImagePreProcess.cpp
//  ImageRetrievalEngine
//
//  Created by XuRui on 15-3-21.
//  Copyright (c) 2015Äê XuRui. All rights reserved.
//
#include "ImagePreProcess.h"

void ImagePreProcess::HSVimgQuantization(const cv::Mat& hsv_img, cv::Mat& quantized_img) {
    std::vector<cv::Mat> splited;
    cv::split(hsv_img.clone(), splited);
    cv::Mat img_H = splited[0];
    cv::Mat img_S = splited[1];
	cv::Mat img_V = splited[2];
	int quantized_H = 0;
	int quantized_S = 0;
	int quantized_V = 0;
	quantized_img.create(hsv_img.rows, hsv_img.cols, CV_8UC1);
	for (int i = 0; i < hsv_img.rows; ++i) {
		for (int j = 0; j < hsv_img.cols; ++j) {
			if (158 <= img_H.at<uchar>(i,j) || img_H.at<uchar>(i,j) <= 10) {
				quantized_H = 0;
			} else if (11 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 20) {
				quantized_H = 1;
			} else if (21 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 38) {
				quantized_H = 2;
			} else if (39 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 77) {
				quantized_H = 3;
			} else if (78 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 95) {
				quantized_H = 4;
			} else if (96 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 135) {
				quantized_H = 5;
			} else if (136 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 147) {
				quantized_H = 6;
			} else if (148 <= img_H.at<uchar>(i,j) && img_H.at<uchar>(i,j) <= 157) {
				quantized_H = 7;
			} if (0 <= img_S.at<uchar>(i,j) && img_S.at<uchar>(i,j) <= 51) {
				quantized_S = 0;
			} else if (52 <= img_S.at<uchar>(i,j) && img_S.at<uchar>(i,j) <= 178) {
				quantized_S = 1;
			} else if (179 <= img_S.at<uchar>(i,j) && img_S.at<uchar>(i,j) <= 255) {
				quantized_S = 2;
			} if (0 <= img_V.at<uchar>(i,j) && img_V.at<uchar>(i,j) <= 51) {
				quantized_V = 0;
			} else if (52 <= img_V.at<uchar>(i,j) && img_V.at<uchar>(i,j) <= 178) {
				quantized_V = 1;
			} else if (179 <= img_V.at<uchar>(i,j) && img_V.at<uchar>(i,j) <= 255) {
				quantized_V = 2;
			}
			quantized_img.at<uchar>(i,j) = 9 * quantized_H + 3 * quantized_S + quantized_V;
		}
	}
}

//resize image
void ImagePreProcess::ImgResize(const cv::Mat& img, cv::Mat& resized_img,const int obj_len) 
{
	int min_len = min(img.rows, img.cols);
	double s = double(obj_len) / min_len;
	resize(img, resized_img, cv::Size(cvRound(s * img.cols), cvRound(s * img.rows)));
}

void ImagePreProcess::imgRGB_HSVQuantization(const Mat &img_rgb, std::vector<int> &code)
{
	if (!img_rgb.data) return;
	if (img_rgb.channels() != 3) return;
	
	Mat img_hsv;
	cvtColor(img_rgb, img_hsv, CV_BGR2HSV);
	std::vector<Mat> splited;
	split(img_hsv, splited);
	Mat h = splited[0];
	Mat s = splited[1];
	Mat v = splited[2];

	Mat res(img_rgb.size(), CV_8UC1);

	for (int i = 0; i < img_rgb.rows; ++i)
	{
		uchar *ph = h.ptr<uchar>(i);
		uchar *ps = s.ptr<uchar>(i);
		uchar *pv = v.ptr<uchar>(i);

		uchar *p_res = res.ptr<uchar>(i);

		for (int j = 0; j < img_rgb.cols; ++j)
		{
			uchar value_h = ph[j];
			uchar value_s = ps[j];
			uchar value_v = pv[j];
			
			uchar quantized_H = 0;
			uchar quantized_S = 0;
			uchar quantized_V = 0;

			if (158 <= value_h || value_h <= 10) {
				quantized_H = 0;
			}
			else if (11 <= value_h && value_h <= 20) {
				quantized_H = 1;
			}
			else if (21 <= value_h && value_h <= 38) {
				quantized_H = 2;
			}
			else if (39 <= value_h && value_h <= 77) {
				quantized_H = 3;
			}
			else if (78 <= value_h && value_h <= 95) {
				quantized_H = 4;
			}
			else if (96 <= value_h && value_h <= 135) {
				quantized_H = 5;
			}
			else if (136 <= value_h && value_h <= 147) {
				quantized_H = 6;
			}
			else if (148 <= value_h && value_h <= 157) {
				quantized_H = 7;
			} 
			
			if (0 <= value_s && value_s <= 51) {
				quantized_S = 0;
			}
			else if (52 <= value_s && value_s <= 178) {
				quantized_S = 1;
			}
			else if (179 <= value_s && value_s <= 255) {
				quantized_S = 2;
			}
			
			if (0 <= value_v && value_v <= 51) {
				quantized_V = 0;
			}
			else if (52 <= value_v && value_v <= 178) {
				quantized_V = 1;
			}
			else if (179 <= value_v && value_v <= 255) {
				quantized_V = 2;
			}
			
			if (quantized_V == 0)
			{
				quantized_S = 0, quantized_H = 0;
			}
			if (quantized_S == 0 && quantized_V == 2) quantized_H = 7;
			
			uchar c = 9 * quantized_H + 3 * quantized_S + quantized_V;
			code.push_back(c);

			p_res[j] = c;

		}
	}

	//normalize(res, res, 0, 255, CV_MINMAX);
	//waitKey(0);
}