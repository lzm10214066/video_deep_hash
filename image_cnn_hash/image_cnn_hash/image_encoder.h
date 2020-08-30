#ifndef _IMAGE_ENCODER_H
#define _IMAGE_ENCODER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "ImagePreProcess.h"
#include "processGif.h"

using namespace std;
using namespace cv;

class ImageEncoder
{
	ImagePreProcess img_prep;
	ProGif pro_gif;
	int obj_len;


	inline string getExten(string &filename) const
	{
		int n = filename.find_last_of('.');
		int m = filename.find_last_not_of(' ');
		return filename.substr(n + 1,m-n);
	}

public:
	ImageEncoder(int _obj_len = 8) :obj_len(_obj_len) {};
	string meanHash(const Mat &img);
	vector<string> meanHashs(const vector<Mat> &imgs);
	string DCTHash(const Mat &img);
	void encoder2vetor(const vector<string> &img_path, vector<vector<int>> &img_codes);
	int imageHashTest_folder(const string &folder, vector<string> & hashs, double &total);
	vector<string> videoHash(string &video_path);
	inline int HanmingDistance(string &str1, string &str2) const
	{
		if ((str1.size() != obj_len*obj_len) || (str2.size() != obj_len*obj_len))
			return -1;
		int difference = 0;
		for (int i = 0; i<obj_len*obj_len; i++)
		{
			if (str1[i] != str2[i])
				difference++;
		}
		return difference;
	}
};


#endif