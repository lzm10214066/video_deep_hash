#include "image_encoder.h"
#include <fstream>
#include "readPath.h"

#ifdef _DEBUG
#define DEBUG_CLIENTBLOCK new( _CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif  // _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new DEBUG_CLIENTBLOCK
#endif  // _DEBUG

void ImageEncoder::encoder2vetor(const vector<string> &img_path, vector<vector<int>> &img_codes)
{
	for (int i = 0; i != img_path.size();++i)
	{
		string tmp = img_path[i];
		Mat img=imread(tmp);
		if (!img.data)
		{
			ProGif pg(1);
			vector<Mat> gif_frames;
			pg.proGif(tmp, gif_frames);
			if(!gif_frames.empty()) img = gif_frames[0];
		}
		if (!img.data)
		{
			cout << "can not read image" << endl;
			continue;
		}

		Mat img_pro;
		img_prep.ImgResize(img, img_pro, obj_len);
		vector<int> code;
		img_prep.imgRGB_HSVQuantization(img_pro, code);
		img_codes.push_back(code);
	}


//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	/*int *c = new int[10];
	vector<int> a;
	a.push_back(1);*/
}

string ImageEncoder::meanHash(const Mat &src)
{
	string rst(obj_len*obj_len, '\0');
	//Mat crop = Mat(src,Rect(src.cols*1/4,src.rows*1/4,src.cols*1/2,src.rows*1/2)).clone();
	Mat crop;
	int w = src.cols;
	int h = src.rows;
	if (h<w)
	{
		int offset = h * 1 / 7;
		int d = (w - h) / 2;
		crop = Mat(src, Rect(d+offset, offset, h-offset, h-offset)).clone();
	}
	else
	{
		int offset = w * 1 / 7;
		int d = (-w + h) / 2;
		crop = Mat(src, Rect(0+offset, d+offset, w-offset, w-offset)).clone();
	}
	Mat img;
	if (crop.channels() == 3)
		cvtColor(crop, img, CV_BGR2GRAY);
	else
		img = crop.clone();
	/*第一步，缩小尺寸。
	将图片缩小到8x8的尺寸，总共64个像素,去除图片的细节*/

	resize(img, img, Size(obj_len, obj_len));
	/* 第二步，简化色彩(Color Reduce)。
	将缩小后的图片，转为64级灰度。*/
	Mat tmp_m, tmp_sd;
	meanStdDev(img, tmp_m, tmp_sd);
	double sd = tmp_sd.at<double>(0, 0);
	if (sd < 5) return string();

	uchar *pData;
	for (int i = 0; i<img.rows; i++)
	{
		pData = img.ptr<uchar>(i);
		for (int j = 0; j<img.cols; j++)
		{
			pData[j] = pData[j] / 4;
		}
	}

	/* 第三步，计算平均值。
	计算所有64个像素的灰度平均值。*/
	int average = mean(img).val[0];

	/* 第四步，比较像素的灰度。
	将每个像素的灰度，与平均值进行比较。大于或等于平均值记为1,小于平均值记为0*/
	Mat mask = (img >= (uchar)average);

	/* 第五步，计算哈希值。*/
	int index = 0;
	for (int i = 0; i<mask.rows; i++)
	{
		pData = mask.ptr<uchar>(i);
		for (int j = 0; j<mask.cols; j++)
		{
			if (pData[j] == 0)
				rst[index++] = '0';
			else
				rst[index++] = '1';
		}
	}
	return rst;
}

string ImageEncoder::DCTHash(const Mat &src)
{
	Mat img, dst;
	string rst(64, '\0');
	double dIdex[64];
	double mean = 0.0;
	int k = 0;

	//Mat crop = Mat(src, Rect(src.cols * 1 / 4, src.rows * 1 / 4, src.cols * 1 / 2, src.rows * 1 / 2)).clone();
	Mat crop;
	int w = src.cols;
	int h = src.rows;
	if (h<w)
	{
		int offset = h * 1 / 7;
		int d = (w - h) / 2;
		crop = Mat(src, Rect(d + offset, offset, h - offset, h - offset)).clone();
	}
	else
	{
		int offset = w * 1 / 7;
		int d = (-w + h) / 2;
		crop = Mat(src, Rect(0 + offset, d + offset, w - offset, w - offset)).clone();
	}
	if (crop.channels() == 3)
	{
		cvtColor(crop, img, CV_BGR2GRAY);
		img = Mat_<double>(img);
	}
	else
	{
		img = Mat_<double>(img);
	}



	/* 第一步，缩放尺寸*/
	resize(img, img, Size(8, 8));

	Mat tmp_m, tmp_sd;
	meanStdDev(img, tmp_m, tmp_sd);
	double sd = tmp_sd.at<double>(0, 0);
	if (sd < 5) return string();

	/* 第二步，离散余弦变换，DCT系数求取*/
	dct(img, dst);

	/* 第三步，求取DCT系数均值（左上角8*8区块的DCT系数）*/
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j)
		{
			dIdex[k] = dst.at<double>(i, j);
			mean += dst.at<double>(i, j) / 64;
			++k;
		}
	}

	/* 第四步，计算哈希值。*/
	for (int i = 0; i<64; ++i)
	{
		if (dIdex[i] >= mean)
		{
			rst[i] = '1';
		}
		else
		{
			rst[i] = '0';
		}
	}
	return rst;
}

vector<string> ImageEncoder::videoHash(string &video_path)
{
	ProGif pg(40);
	vector<Mat> gif_frames;
	pg.proGif(video_path, gif_frames);

	//ImageEncoder encoder;
	vector<string> hashs;
	hashs.reserve(gif_frames.size());

	for (int i = 2; i < (int)gif_frames.size(); ++i)
	{
		string h = meanHash(gif_frames[i]);
		//string h = encoder.DCTHash(gif_frames[i]);
		if (!h.empty()) {
			cout << h << endl;
			hashs.push_back(h);
		}
	}
	return hashs;
}

vector<string> ImageEncoder::meanHashs(const vector<Mat> &imgs)
{
	vector<string> res;
	for (auto img : imgs)
	{
		string tmp = meanHash(img);
		res.push_back(tmp);
	}
	return res;
}

int ImageEncoder::imageHashTest_folder(const string &folder, vector<string> &hashs_out, double &total)
{
	vector<string> paths = getFiles(folder, true);
	vector<Mat> imgs;
	for (auto p : paths)
	{
		Mat img = imread(p);
		imgs.push_back(img);
	}
	vector<string> hashs = meanHashs(imgs);
	int error_count = 0;
	for (int i = 0; i < hashs.size(); ++i)
	{
		for (int j = i + 1; j < hashs.size(); ++j)
		{
			int d = HanmingDistance(hashs[i], hashs[j]);
			if (d > 1)
			{
				cout << d << "  ";
				error_count += 1;
			}

		}
	}
	total = hashs.size()*(hashs.size() - 1) / 2;
	hashs_out.push_back(hashs[0]);
	hashs_out.push_back(hashs[hashs.size() - 1]);  //save the first and last one
	return error_count;
}