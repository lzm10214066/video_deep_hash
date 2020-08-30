#include "CaffeFeature.h"

CaffeFeature::CaffeFeature() {}

CaffeFeature::CaffeFeature(const std::string &model_file, const std::string &trained_file):vector_size(2048)
{
//#ifdef CPU_ONLY
//	Caffe::set_mode(Caffe::CPU);
//	std::cout << "set CPU" << std::endl;
//#else
//	Caffe::set_mode(Caffe::GPU);
//	std::cout << "set GPU" << std::endl;
//#endif

	//std::shared_ptr<Net<float>> net;
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}

CaffeFeature::~CaffeFeature() {}

void CaffeFeature::extractFeatureByOut(const cv::Mat &img,vector<float> &feature_vector)
{
	Mat img_data=Preprocess(img);
	CV_Assert(img_data.size() == input_geometry_);
	CV_Assert(img_data.channels() == num_channels_);
	feature_vector=PredictByOut(img_data);
}

void CaffeFeature::extractFeatureByOut(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector)
{
	//CV_Assert(img[0].size() == input_geometry_);
	//CV_Assert(img[0].channels() == num_channels_);
	vector<Mat> img_datas(imgs.size());
	for (int i = 0; i < imgs.size(); ++i)
	{
		img_datas[i] = Preprocess(imgs[i]);
	}
	
	feature_vector = PredictByOut(img_datas);
}

Mat CaffeFeature::Preprocess(const cv::Mat &img)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample.convertTo(sample_float, CV_32FC3);
	else
		sample.convertTo(sample_float, CV_32FC1);

	sample_float = (sample_float - 127.5) / 128.;

	return resizeAndCrop(sample_float);
}

/*
 * Predict function input is a image without crop
 * the reshape of input layer is image's height and width
 */
vector<float> CaffeFeature::PredictByOut(const cv::Mat& img)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(img, &input_channels);
	net->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/*
 * Predict(const std::vector<cv::Mat> imgs, int i) function
 * used to input is a group of image with crop from original image
 * the reshape of input layer of net is the number, channels, height and width of images.
 */
vector<vector<float>> CaffeFeature::PredictByOut(const std::vector<cv::Mat> imgs)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_,
		input_geometry_.height, input_geometry_.width);
	int num = input_layer->num();
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(imgs, &input_channels);

	net->Forward();

	vector<vector<float>> features(imgs.size());
	Blob<float>* output_layer = net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();

	/* Copy the output layer to a std::vector */
	//You can also try to use the blob_by_name()
	for (int i = 0; i < imgs.size(); ++i)
	{
		int s = i*vector_size;
		features[i] = vector<float>(begin + s, begin + s + vector_size);
	}
	return features;
}

void CaffeFeature::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int j = 0; j < input_layer->channels(); ++j)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}

	//cv::Mat sample_normalized;
	//cv::subtract(img, mean_[i], img);
	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(img, *input_channels);

}

/*
 * WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) function
 * used to write the separate BGR planes directly to the input layer of the network
 */
void CaffeFeature::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels)
{
	Blob<float> *input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float *input_data = input_layer->mutable_cpu_data();

	for (int j = 0; j < num; j++) {
		//std::vector<cv::Mat> *input_channels;
		for (int k = 0; k < input_layer->channels(); ++k) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
		cv::Mat img = imgs[j];
		cv::split(img, *input_channels);
		input_channels->clear();
	}
}


void CaffeFeature::Padding(std::vector<cv::Rect>& bounding_box, int img_w, int img_h)
{
	for (int i = 0; i < bounding_box.size(); i++)
	{
		bounding_box[i].x = (bounding_box[i].x > 0) ? bounding_box[i].x : 0;
		bounding_box[i].y = (bounding_box[i].y > 0) ? bounding_box[i].y : 0;
		bounding_box[i].width = (bounding_box[i].x + bounding_box[i].width < img_w) ? bounding_box[i].width : img_w;
		bounding_box[i].height = (bounding_box[i].y + bounding_box[i].height < img_h) ? bounding_box[i].height : img_h;
	}
}

cv::Mat CaffeFeature::crop(cv::Mat &img, cv::Rect& rect)
{
	cv::Rect rect_old = rect;

	//    if(rect.width > rect.height)
	//    {
	//        int change_to_square = rect.width - rect.height;
	//        rect.height += change_to_square;
	//        rect.y -= change_to_square * 0.5;
	//    }
	//    else
	//    {
	//        int change_to_square = rect.height - rect.width;
	//        rect.width += change_to_square;
	//        rect.x -= change_to_square * 0.5;
	//    }

	cv::Rect padding;

	if (rect.x < 0)
	{
		padding.x = -rect.x;
		rect.x = 0;
	}
	if (rect.y < 0)
	{
		padding.y = -rect.y;
		rect.y = 0;
	}
	if (img.cols < (rect.x + rect.width))
	{
		padding.width = rect.x + rect.width - img.cols;
		rect.width = img.cols - rect.x;
	}
	if (img.rows < (rect.y + rect.height))
	{
		padding.height = rect.y + rect.height - img.rows;
		rect.height = img.rows - rect.y;
	}
	if (rect.width < 0 || rect.height < 0)
	{
		rect = cv::Rect(0, 0, 0, 0);
		padding = cv::Rect(0, 0, 0, 0);
	}
	cv::Mat img_cropped = img(rect);
	if (padding.x || padding.y || padding.width || padding.height)
	{
		cv::copyMakeBorder(img_cropped, img_cropped, padding.y, padding.height, padding.x, padding.width, cv::BORDER_CONSTANT, cv::Scalar(0));
		//here, the rect should be changed
		rect.x -= padding.x;
		rect.y -= padding.y;
		rect.width += padding.width + padding.x;
		rect.width += padding.height + padding.y;
	}

	//    cv::imshow("crop", img_cropped);
	//    cv::waitKey(0);

	return img_cropped;
}

/*
* Predict function input is a image without crop
* the reshape of input layer is image's height and width
*/
vector<float> CaffeFeature::PredictByBlobName(const cv::Mat& img, const string &blob_name)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(img, &input_channels);
	net->Forward();

	CHECK(net->has_blob(blob_name))
		<< "Unknown feature blob name " << blob_name;
	const boost::shared_ptr<Blob<float> > feature_blob =
		net->blob_by_name(blob_name);
	/* Copy the output layer to a std::vector */
	const float* begin = feature_blob->cpu_data();
	//const float* end = begin + feature_blob->channels();
	const float* end = begin + feature_blob->count();
	return std::vector<float>(begin, end);
}

/*
* Predict(const std::vector<cv::Mat> imgs, int i) function
* used to input is a group of image with crop from original image
* the reshape of input layer of net is the number, channels, height and width of images.
*/
vector<vector<float>> CaffeFeature::PredictByBlobName(const std::vector<cv::Mat> imgs, const string &blob_name)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_,
		input_geometry_.height, input_geometry_.width);
	int num = input_layer->num();
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(imgs, &input_channels);

	net->Forward();

	CHECK(net->has_blob(blob_name))
		<< "Unknown feature blob name " << blob_name;
	const boost::shared_ptr<Blob<float> > feature_blob =
		net->blob_by_name(blob_name);



	vector<vector<float>> features(imgs.size());
	const float* begin = feature_blob->cpu_data();
	//const float* end = begin + feature_blob->channels();
	const float* end = begin + feature_blob->count();
	/* Copy the output layer to a std::vector */
	//You can also try to use the blob_by_name()
	for (int i = 0; i < imgs.size(); ++i)
	{
		int s = i*vector_size;
		features[i] = vector<float>(begin + s, begin + s + vector_size);
	}
	return features;
}

void CaffeFeature::extractFeatureByBlobName(const cv::Mat &img, vector<float> &feature_vector,const string &blob_name)
{
	Mat img_data = Preprocess(img);
	CV_Assert(img_data.size() == input_geometry_);
	CV_Assert(img_data.channels() == num_channels_);
	
	feature_vector = PredictByBlobName(img_data,blob_name);
}

vector<float> CaffeFeature::PredictByBlobName_for_featureMap(const cv::Mat& img, const string &blob_name, vector<int> &shape)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(img, &input_channels);
	net->Forward();

	CHECK(net->has_blob(blob_name))
		<< "Unknown feature blob name " << blob_name;
	const boost::shared_ptr<Blob<float> > feature_blob =
		net->blob_by_name(blob_name);
	/* Copy the output layer to a std::vector */
	const float* begin = feature_blob->cpu_data();
	//const float* end = begin + feature_blob->channels();
	const float* end = begin + feature_blob->count();

	shape=feature_blob->shape();
	return std::vector<float>(begin, end);
}

void CaffeFeature::extractFeatureMapBlobName(const cv::Mat &img, vector<float> &feature_vector, const string &blob_name,vector<int> &shape)
{
	Mat img_data = Preprocess(img);
	CV_Assert(img_data.size() == input_geometry_);
	CV_Assert(img_data.channels() == num_channels_);

	feature_vector = PredictByBlobName_for_featureMap(img_data, blob_name,shape);
}

void CaffeFeature::extractFeatureByBlobName(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector,const string &blob_name)
{
	//CV_Assert(img[0].size() == input_geometry_);
	//CV_Assert(img[0].channels() == num_channels_);
	vector<Mat> img_datas(imgs.size());
	for (int i = 0; i < imgs.size(); ++i)
	{
		img_datas[i] = Preprocess(imgs[i]);
	}

	feature_vector = PredictByBlobName(img_datas,blob_name);
}

void CaffeFeature::extractFeatureByBlobName(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vectors, const string &blob_name,const int batch_size)
{
	if (imgs.empty()) return;
	vector<Mat> img_datas(imgs.size());
	for (int i = 0; i < imgs.size(); ++i)
	{
		img_datas[i] = Preprocess(imgs[i]);
	}

	CV_Assert(img_datas[0].size() == input_geometry_);
	CV_Assert(img_datas[0].channels() == num_channels_);

	for (int i = 0;; ++i)
	{
		int s = i*batch_size;
		if (s >= img_datas.size())
			break;
		int e = (i + 1)*batch_size;
		if (e >= img_datas.size())
			e = img_datas.size();
		if (s == e) break;
		//std::cout << "s,e: " << s << e << std::endl;
		vector<Mat> imgs_batch(img_datas.begin() + s, img_datas.begin() + e);
		vector<vector<float>> features_batch= PredictByBlobName(imgs_batch, blob_name);
		feature_vectors.insert(feature_vectors.end(), features_batch.begin(), features_batch.end());
	}
}

Mat CaffeFeature::resizeAndCrop(const Mat &img)
{
	int length = input_geometry_.height;
	int h = img.rows;
	int w = img.cols;
	int min_l = min(h, w);
	double s = double(length) / min_l;
	Size new_size = h > w ? Size(length, round(h*s)) : Size(round(w*s), length);
	Mat tmp;
	resize(img, tmp, new_size);
	int n_w = new_size.width;
	int n_h = new_size.height;
	if (n_h > n_w)
	{
		int dh = (n_h - n_w) / 2;
		Rect r = Rect(0, dh, length, length);
		return Mat(tmp, r).clone();
	}
	else
	{
		int dw= (n_w - n_h) / 2;
		Rect r = Rect(dw, 0, length, length);
		return Mat(tmp, r).clone();
	}
}
