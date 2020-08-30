
#ifndef _CaffeFeature_H
#define _CaffeFeature_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;
using namespace cv;

class CaffeFeature {

public:

	CaffeFeature();
	CaffeFeature(const std::string &model_file, const std::string &trained_file);
    ~CaffeFeature();

	void extractFeatureByOut(const cv::Mat &img, vector<float> &feature_vector);
	void extractFeatureByOut(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector);

	void extractFeatureByBlobName(const cv::Mat &img, vector<float> &feature_vector, const string &blob_name);
	void extractFeatureByBlobName(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector, const string &blob_name);
	void extractFeatureByBlobName(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector, const string &blob_name, const int batch_size);
	void extractFeatureMapBlobName(const cv::Mat &img, vector<float> &feature_vector, const string &blob_name, vector<int> &shape);

private:
	Mat Preprocess(const cv::Mat &img);
   
	vector<float> PredictByOut(const cv::Mat& img);
	vector<vector<float>> PredictByOut(const std::vector<cv::Mat> imgs);

	vector<float> PredictByBlobName(const cv::Mat& img, const string &blob_name);
	vector<vector<float>> PredictByBlobName(const std::vector<cv::Mat> imgs, const string &blob_name);

	vector<float> PredictByBlobName_for_featureMap(const cv::Mat& img, const string &blob_name, vector<int> &shape);


    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels);
    void Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h);
    cv::Mat crop(cv::Mat &img, cv::Rect& rect);
	Mat resizeAndCrop(const Mat &img);
    //param for net
    std::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
	int vector_size;

};


#endif //_CaffeFeature_H
