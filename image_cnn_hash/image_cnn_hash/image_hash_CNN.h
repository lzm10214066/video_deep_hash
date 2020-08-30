#ifndef _IMAGE_HASH_CNN_H
#define _IMAGE_HASH_CNN_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "caffeFeature.h"
#include "readPath.h"

using namespace std;
using namespace cv;

class CNNImageHash
{
	
	int obj_len;
	Ptr<CaffeFeature> fea_extractor;
	inline string getExten(string &filename) const
	{
		int n = filename.find_last_of('.');
		int m = filename.find_last_not_of(' ');
		return filename.substr(n + 1,m-n);
	}

public:
	CNNImageHash(int obj_len_ = 8);
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

	string getImgHash(const Mat &imgs);
	vector<string> getImgsHash(const vector<Mat> &imgs);
	int imageHashTest_folder(const string &folder, vector<string> & hashs, double &total);
	Mat visualize_featureMap(const Mat &src,const string blob_name, int channel_idx);
};


#endif