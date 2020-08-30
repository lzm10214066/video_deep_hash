#include "image_hash_CNN.h"
#include "processGif.h"

CNNImageHash::CNNImageHash(int obj_len_) :obj_len(obj_len_)
{
	string model_file = "models/beta_16_ip.prototxt";
	string weights_file = "models/train_iter_10000.caffemodel";

	fea_extractor = new CaffeFeature(model_file, weights_file);
}

string CNNImageHash::getImgHash(const Mat &src)
{
	vector<float> tmp;
	fea_extractor->extractFeatureByBlobName(src, tmp, "ip2");
	
	/*double sum = 0;
	for (auto h : tmp){ sum += h; }
	double mean = sum/64;
	string res;
	for (auto h : tmp)
	{
		if (h >= mean) res += "1";
		else res += "0";
	}*/

	string res;
	for (auto h : tmp)
	{
		if (h >= 0) res += "1";
		else res += "0";
	}
	return res;
}

vector<string> CNNImageHash::getImgsHash(const vector<Mat> &imgs)
{
	vector<string> res;
	for (auto img : imgs)
	{
		string tmp = getImgHash(img);
		res.push_back(tmp);
	}
	return res;
}
 int CNNImageHash::imageHashTest_folder(const string &folder, vector<string> & hashs_out, double &total)
{
	vector<string> paths = getFiles(folder, true);
	vector<Mat> imgs;
	for (auto p : paths)
	{
		Mat img = imread(p);
		imgs.push_back(img);
	}
	vector<string> hashs = getImgsHash(imgs);
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
	hashs_out.push_back(hashs[hashs.size()-1]);  //save the first and last one
	return error_count;
}

 Mat CNNImageHash::visualize_featureMap(const Mat &src, const string blob_name, int channel_idx)
 {
	 vector<float> tmp;
	 vector<int> shape;
	 fea_extractor->extractFeatureMapBlobName(src, tmp, blob_name,shape);
	 int h_w = shape[2] * shape[3];
	 vector<float> res(tmp.begin() + channel_idx*h_w, tmp.begin() + (channel_idx + 1)*h_w);
	 normalize(res, res, 255, 0, NORM_MINMAX);
	 Mat res_tmp(res);
	 Mat res_mat=res_tmp.reshape(1,shape[2]);
	 return res_mat;
 }

 vector<string> CNNImageHash::videoHash(string &video_path)
 {
	 ProGif pg(40);
	 vector<Mat> gif_frames;
	 pg.proGif(video_path, gif_frames);

	 //ImageEncoder encoder;
	 vector<string> hashs;
	 hashs.reserve(gif_frames.size());

	 for (int i = 2; i < (int)gif_frames.size(); ++i)
	 {
		 string h = getImgHash(gif_frames[i]);
		 //string h = encoder.DCTHash(gif_frames[i]);
		 if (!h.empty()) {
			 cout << h << endl;
			 hashs.push_back(h);
		 }
	 }
	 return hashs;
 }