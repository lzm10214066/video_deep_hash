#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <iostream>  

#include "video_process.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	
	/*string folder_in = argv[1];
	string folder_out = argv[2];
	int num_sample_frames = atoi(argv[3]);*/

	double t = (double)getTickCount();

	string fold_in = argv[1];
	string fold_out = argv[2];
	int num_thread = atoi(argv[3]);
	processVideos2ImgsThreads(fold_in,fold_out,num_thread);

	double time = ((double)getTickCount() - t) / getTickFrequency()*1000;
	cout <<"\ntime passed in ms : " << time << endl;


	return 0;
}

