#ifndef _VIDEO_PROCESS_H
#define _VIDEO_PROCESS_H

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include <iostream>  
#include <string>
#include <fstream>
#include <thread>

#include "readPath.h"

using namespace std;
using namespace cv;

class Similar_Images_Video
{
	int obj_len;

	void cropImage(vector<Mat> &imgs, const int &new_height, const int &new_width);
	bool isSimilar(const Mat &img1, const Mat &img2);
	string meanHash(const Mat &src);
	string DCTHash(const Mat &src);
	vector<string> imagesHash(vector<Mat> &imgs);
	vector<Mat> dataAugmentation(const vector<Mat> &src);
	vector<Mat> removeDiffrence(vector<Mat> &imgs);
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

public:
	Similar_Images_Video(int _obj_len = 8) :obj_len(_obj_len) {};
	void sample_similar_images_from_video(const string &video_path,
		const string &folder_out,
		const int new_height=128,
		const int new_width=128,
		const int step=200,
		const int num_per_sample=5,
		const int num_frame_limit=180*25);
	void sample_images_from_video(
		const string &video_path,
		const string &folder_out,
		const int new_height = 299,
		const int new_width = 299,
		const int time_limit=2,  //minute
		const int num_frame_limit = 10);
};
void processVideos2ImgsThreads(const string &folder_in, const string &folder_out, const int num_thread);

#endif