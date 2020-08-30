#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"
#include "gif_ffmpeg.h"

using namespace std;
using namespace cv;

class ProGif
{
	int max_num;
	double th;
	int timeStep;
	double simiTwoImages(Mat &imga, Mat &imgb);
	void duplicateFilter(vector<Mat> &frames, double th);
	int  gif2frames(const char* filename, const string &folderName);
	int  pickGif2Mats(const char* filename, int max_num, vector<Mat> &gif_frames);
public:
	ProGif(int _max_num = 20, double _th=0.8, int _timeStep=0);
	int proGif(string &fileName, vector<Mat> &frames);
	
};


