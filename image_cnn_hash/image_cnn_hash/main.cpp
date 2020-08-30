#include <iostream>
#include "opencv2\opencv.hpp"
#include "readPath.h"
#include "image_hash_CNN.h"
#include "image_encoder.h"
#include "glog\logging.h"

using namespace std;
using namespace cv;

#define CNN_test true
#define mean_test false
#define video_test_mean false
#define video_test_cnn false



int main(int argc, char** argv)
{
	caffe::GlobalInit(&argc, &argv);

	if (CNN_test)
	{
		CNNImageHash cnn_img_hash;

		//string root = "E:\\data_no_dataArgu_part";
		string root = "test";
		vector<string> folders = getFiles(root, true);
		double total_pos = 0;
		int e_pos = 0;
		vector<vector<string>> hashs_total;
		for (auto f : folders)
		{
			cout << f << endl;
			vector<string> hashs_per_folder;
			double total_per_folder;
			int e = cnn_img_hash.imageHashTest_folder(f, hashs_per_folder, total_per_folder);
			if (e > 0) cout << "watch out the folder: " << e << endl;
			total_pos += total_per_folder;
			e_pos += e;
			hashs_total.push_back(hashs_per_folder);
		}
		cout << "error for similar images;  " << e_pos << " " << total_pos << " " << e_pos / total_pos << endl;

		int e_neg = 0;
		for (int i = 0; i < hashs_total.size(); ++i)
		{
			string h0 = hashs_total[i][0];
			string h1 = hashs_total[i][1];
			for (int j = i + 1; j < hashs_total.size(); ++j)
			{
				string h00 = hashs_total[j][0];
				string h11 = hashs_total[j][1];
				int d = cnn_img_hash.HanmingDistance(h0, h00);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = cnn_img_hash.HanmingDistance(h0, h11);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = cnn_img_hash.HanmingDistance(h1, h00);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = cnn_img_hash.HanmingDistance(h1, h11);
				if (d < 2) { e_neg += 1; cout << d << "  "; }
				cout << d << "  ";
			}
			cout << "\n";
		}
		double total_neg = hashs_total.size()*hashs_total.size() - 1 / 2 * 4;
		cout << "error for non-similar images;  " << e_neg << " " << total_neg << " " << e_neg / total_neg << endl;

		/*Mat img = imread("1.png");
		cnn_img_hash.visualize_featureMap(img, "conv5", 0);*/
	}
	
	if (mean_test)
	{
		ImageEncoder encoder;
		//string root = "E:\\data_no_dataArgu_part";
		string root = "test";
		vector<string> folders = getFiles(root, true);
		double total_pos = 0;
		int e_pos = 0;
		vector<vector<string>> hashs_total;
		for (auto f : folders)
		{
			cout << f << endl;
			vector<string> hashs_per_folder;
			double total_per_folder;
			int e = encoder.imageHashTest_folder(f, hashs_per_folder, total_per_folder);
			if (e > 0) cout << "watch out the folder: " << e << endl;
			total_pos += total_per_folder;
			e_pos += e;
			hashs_total.push_back(hashs_per_folder);
		}
		cout << "error for similar images;  " << e_pos << " " << total_pos << " " << e_pos / total_pos << endl;

		int e_neg = 0;
		for (int i = 0; i < hashs_total.size(); ++i)
		{
			string h0 = hashs_total[i][0];
			string h1 = hashs_total[i][1];
			for (int j = i + 1; j < hashs_total.size(); ++j)
			{
				string h00 = hashs_total[j][0];
				string h11 = hashs_total[j][1];
				int d = encoder.HanmingDistance(h0, h00);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = encoder.HanmingDistance(h0, h11);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = encoder.HanmingDistance(h1, h00);
				if (d < 2) { e_neg += 1; cout << d << "  "; }

				d = encoder.HanmingDistance(h1, h11);
				if (d < 2) { e_neg += 1; cout << d << "  "; }
				cout << d << "  ";
			}
			cout << "\n";
		}
		double total_neg = hashs_total.size()*hashs_total.size() - 1 / 2 * 4;
		cout << "error for non-similar images;  " << e_neg << " " << total_neg << " " << e_neg / total_neg << endl;
	}

	if (video_test_mean)
	{
		string video_0 = "1.mp4";
		string video_1 = "2.mp4";
		ImageEncoder encoder;

		cout << video_0 << endl;
		vector<string> hash_0 = encoder.videoHash(video_0);
		cout << video_1 << endl;
		vector<string> hash_1 = encoder.videoHash(video_1);

		int count = 0;
		for (int i = 0; i < hash_0.size(); ++i)
		{
			if (hash_0[i].empty()) continue;
			for (int j = 0; j < hash_1.size(); ++j)
			{
				if (hash_1[j].empty()) continue;
				int d = encoder.HanmingDistance(hash_0[i], hash_1[j]);
				if (d < 2)
				{
					cout << d << endl;
					count++;
					break;
				}
			}
		}
		cout << "count: " << count << endl;
	}
	if (video_test_cnn)
	{
		string video_0 = "0.mp4";
		string video_1 = "00.mp4";
		CNNImageHash encoder;

		cout << video_0 << endl;
		vector<string> hash_0 = encoder.videoHash(video_0);
		cout << video_1 << endl;
		vector<string> hash_1 = encoder.videoHash(video_1);

		int count = 0;
		for (int i = 0; i < hash_0.size(); ++i)
		{
			if (hash_0[i].empty()) continue;
			for (int j = 0; j < hash_1.size(); ++j)
			{
				if (hash_1[j].empty()) continue;
				int d = encoder.HanmingDistance(hash_0[i], hash_1[j]);
				if (d < 2)
				{
					cout << d << endl;
					count++;
					break;
				}
			}
		}
		cout << "count: " << count << endl;
	}

	return 0;
}

