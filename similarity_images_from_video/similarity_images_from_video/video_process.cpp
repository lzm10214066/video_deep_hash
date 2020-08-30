#include "video_process.h"
#include "preProcessImage.h"

void Similar_Images_Video::sample_similar_images_from_video(const string &video_path,
	const string &folder_out,
	const int new_height,
	const int new_width,
	const int step,
	const int num_per_sample,
	const int num_frames_limit)
{
	//if the video has been visited
	string videoName = getFileName(video_path);
	string dst_classFolder = folder_out + "/" + videoName + "_" + to_string(0);
	if (_access(dst_classFolder.c_str(), 0) != -1)
	{
		cout << "the video has been visited" << endl;
		return;
	}

	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		cout << "fail to open!" << endl; return;
	}
	long frameToStart = 50;
	long totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (totalFrameNumber < frameToStart + step + num_per_sample + 5)
	{
		cout << "the video is too short" << endl; return;
	}

	cap.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	
	int i = 0;
	bool f = true;
	vector<vector<Mat>> frames;

	//sample frames in every step
	while(f && i<num_frames_limit)
	{
		vector<Mat> frames_per_sample;
		for (int j = 0; j < num_per_sample; ++j)
		{
			cap >> image;
			if (image.cols == 0 || image.rows == 0) break;
			frames_per_sample.push_back(image.clone());
		}
		vector<Mat> tmp = removeDiffrence(frames_per_sample);
		if (tmp.size() >= 2)
		{
			//vector<Mat> trans_imgs = dataAugmentation(tmp);
			vector<Mat> trans_imgs = tmp;
			cropImage(trans_imgs, new_height, new_width);
			frames.push_back(trans_imgs);
		}
		int temp_step = step;
		while (temp_step--)
		{
			f = cap.grab();
			if (f == false) break;
		}
		i++;
	}

	//remove duplicated images
	//get hashs
	if (frames.size() > 2)
	{
		vector<bool> remove(frames.size(), false);
		vector<string> hashs(frames.size());
		for (int i = 0; i < frames.size(); ++i)
		{
			hashs[i] = meanHash(frames[i][0]);
		}
		for (int j = 0; j < hashs.size(); ++j)
		{
			if (remove[j] == true) continue;
			for (int k = j + 1; k < hashs.size(); ++k)
			{
				if (remove[k] == true) continue;
				int d = HanmingDistance(hashs[j], hashs[k]);
				cout << d << endl;
				if (HanmingDistance(hashs[j], hashs[k]) < 20)
					remove[k] = true;
			}
		}
		vector<vector<Mat>> tmp;
		for (int i = 0; i < remove.size(); ++i)
		{
			if (remove[i] == false)
				tmp.push_back(frames[i]);
		}
		frames = tmp;
	}
	for (int i = 0; i < frames.size(); ++i)
	{
		string dst_classFolder = folder_out + "/" + videoName+"_"+to_string(i);
		if (_access(dst_classFolder.c_str(), 0) == -1)
		{
			cout << dst_classFolder << " is not existing" << endl;
			int flag = _mkdir(dst_classFolder.c_str());
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}
		for (int j = 0; j < frames[i].size(); ++j)
		{
			string dst_name = dst_classFolder + "/"+ to_string(j) + ".png";
			imwrite(dst_name, frames[i][j]);
		}
	}
}

void Similar_Images_Video::sample_images_from_video(
	const string &video_path,
	const string &folder_out,
	const int new_height,
	const int new_width,
	const int time_limit,  //minute
	const int num_frames_limit)
{

	//if the video has been visited
	string videoName = getFileName(video_path);
	string dst_classFolder = folder_out + "/" + videoName;
	if (_access(dst_classFolder.c_str(), 0) != -1)
	{
		cout << "the video has been visited" << endl;
		return;
	}

	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		cout << "fail to open!" << endl; return;
	}

	double fps = cap.get(CV_CAP_PROP_FPS);
	long frameToStart = fps * 10;
	long totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (totalFrameNumber < frameToStart + 5) { cout << "the video is too short" << endl; return; }

	{
		cout << dst_classFolder << " is not existing" << endl;
		int flag = _mkdir(dst_classFolder.c_str());
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
	}

	cap.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	const int step = time_limit * 60 * fps / 10;  //optional

	int i = 0;
	vector<Mat> frames;
	long total_limit = time_limit * 60 * fps+ frameToStart;
	bool f = true;
	//sample frames in every step
	while (f && i<num_frames_limit)
	{
		cap >> image;
		if (image.cols == 0 || image.rows == 0) break;
		frames.push_back(image.clone());
		i++;

		int temp_step = step-1;
		while (temp_step--)
		{
			f = cap.grab();
			if (f == false) break;
		}
	}
	cropImage(frames, new_height, new_width);
	for (int j = 0; j < frames.size(); ++j)
	{
		string dst_name = dst_classFolder + "/" + to_string(j) + ".jpg";
		imwrite(dst_name, frames[j]);
	}

}

vector<Mat>Similar_Images_Video::dataAugmentation(const vector<Mat> &src)
{
	ImagePreProcess imagePre;
	vector<Mat> trans_imgs;
	for (auto img : src)
	{
		Mat img_noise = img.clone();
		imagePre.addGaussianNoise(img_noise, 0, 8);
		trans_imgs.push_back(img_noise);

		Mat img_blur = img.clone();
		medianBlur(img_blur, img_blur, 5);
		trans_imgs.push_back(img_blur);

		Mat img_resize = img.clone();
		int k = 3;
		double s_r = 1.0 / 4;
		double e_r = 4.;
		for (double r = s_r; r <= e_r;)
		{
			Mat t;
			imagePre.resizeBlur(img_resize, t,r);
			trans_imgs.push_back(t);
			r += (e_r - s_r) / k;
		}
	}
	return trans_imgs;
}

vector<Mat> Similar_Images_Video::removeDiffrence(vector<Mat> &imgs)
{
	if (imgs.size() < 2) return imgs;
	vector<string> hashs = imagesHash(imgs);
	vector<bool> remove(hashs.size(),false);
	for (int i = 1; i < imgs.size(); ++i)
	{
		if (HanmingDistance(hashs[0], hashs[i]) > 8)
			remove[i] = true;
	}
	vector<Mat> res;
	for (int k=0;k<remove.size();++k)
	{
		bool r = remove[k];
		if (r == false)
			res.push_back(imgs[k]);
	}
	return res;
}

void Similar_Images_Video::cropImage(vector<Mat> &imgs, const int &new_height, const int &new_width)
{
	if (imgs.empty()) return;
	int frame_height = imgs[0].rows;
	int frame_width = imgs[0].cols;
	bool h_or_w = frame_height > frame_width ? true : false;

	for (auto &img : imgs)
	{
		if (h_or_w)
		{
			int dh = (frame_height - frame_width) / 2;
			Mat croped_img = img(Range(dh, dh + frame_width), Range::all());
			if (new_height && new_width)
				resize(croped_img, croped_img, Size(new_height, new_width));
			img = croped_img;
		}
		else
		{
			int dw = (-frame_height + frame_width) / 2;
			Mat croped_img = img(Range::all(), Range(dw, dw + frame_height));
			if (new_height && new_width)
				resize(croped_img, croped_img, Size(new_height, new_width));
			img = croped_img;
		}
	}
}

string Similar_Images_Video::meanHash(const Mat &src)
{
	string rst(obj_len*obj_len, '\0');
	int w = src.cols;
	int h = src.rows;
	Mat img;
	if (src.channels() == 3)
		cvtColor(src, img, CV_BGR2GRAY);
	else
		img = src.clone();
	/*第一步，缩小尺寸。
	将图片缩小到8x8的尺寸，总共64个像素,去除图片的细节*/

	resize(img, img, Size(obj_len, obj_len));
	/* 第二步，简化色彩(Color Reduce)。
	将缩小后的图片，转为64级灰度。*/

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

string Similar_Images_Video::DCTHash(const Mat &src)
{
	Mat img, dst;
	string rst(64, '\0');
	double dIdex[64];
	double mean = 0.0;
	int k = 0;

	if (src.channels() == 3)
	{
		cvtColor(src, img, CV_BGR2GRAY);
		img = Mat_<double>(img);
	}
	else
	{
		img = Mat_<double>(src);
	}



	/* 第一步，缩放尺寸*/
	resize(img, img, Size(8, 8));
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

vector<string> Similar_Images_Video::imagesHash(vector<Mat> &imgs)
{
	vector<string> hashs;
	for (auto img : imgs)
	{
		//string hash = meanHash(img);
		string hash = DCTHash(img);
		hashs.push_back(hash);
	}
	return hashs;
}

void processVideo_thread(const vector<string> &video_paths, const int s, const int t, const string &dst_classFolder)
{

	Similar_Images_Video videoImg;
	for (int i = s; i < s + t; ++i)
	{
		string video_path = video_paths[i];
		cout << "i: " << i << "   " << video_path << endl;

		//videoImg.sample_similar_images_from_video(video_path, dst_classFolder,150,150,250,10);
		videoImg.sample_images_from_video(video_path, dst_classFolder);
	}
}

void processVideos2ImgsThreads(const string &folder_in, const string &folder_out, const int num_thread)
{
	vector<string> filePaths = getFiles(folder_in, true);
	vector<string> video_paths;
	for (int i = 0; i < filePaths.size(); ++i)
	{
		string class_path = filePaths[i];
		vector<string> tmp = getFiles(class_path, true);
		video_paths.insert(video_paths.end(), tmp.begin(), tmp.end());
	}
	int total_videos = video_paths.size();
	int videos_per_thread = (int)ceil((double)total_videos / num_thread);
	vector<thread> threads;

	int s = 0;
	for (int i = 0; i < num_thread; ++i)
	{
		int t = min(videos_per_thread, total_videos - s);
		threads.push_back(thread(processVideo_thread, video_paths, s, t, folder_out));
		s += t;
	}

	for (int i = 0; i < threads.size(); i++)
		threads[i].join();

}
