#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include<string>
#include "SlicCudaHost.h"

using namespace std;
using namespace cv;

int main() {
 
	// VideoCapture cap("/home/jimmy/ece508/data/waterfall.avi");


	// Parameters
	int diamSpx = 32;
	int wc = 35;
	int nIteration = 10;
	SlicCuda::InitType initType = SlicCuda::SLIC_SIZE;

	//start segmentation
	Mat frame;
	frame = imread("/home/jimmy/ece508/data/ocean_coast.jpg", IMREAD_COLOR);

	// cap >> frame;

	SlicCuda oSlicCuda;
	oSlicCuda.initialize(frame, diamSpx, initType, wc, nIteration);

	// int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	// int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	// int endFrame = cap.get(CAP_PROP_FRAME_COUNT);
    // double fps = cap.get(CAP_PROP_FPS);

    // VideoWriter video("/home/jimmy/ece508/segs/waterfall.avi", 
	// 	VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width,frame_height));


	int i = 0;
	int endFrame = 1;
	for (i=0; i<endFrame; i++){

		auto t0 = std::chrono::high_resolution_clock::now();
		oSlicCuda.segment(frame);

		auto t1 = std::chrono::high_resolution_clock::now();
		double time = std::chrono::duration<double>(t1-t0).count() ;
		cout << "Frame " << frame.size() <<" "<< i+1 << "/" << endFrame << ", Time: "<< time <<"s"<<endl;
	}

	oSlicCuda.enforceConnectivity();

	// std::cout << labels << endl;
	Mat labels = oSlicCuda.getLabels();
	SlicCuda::displayBound(frame, (float*)labels.data, Scalar(255, 255, 255));
	imwrite("/home/jimmy/ece508/segs/ocean_coast.jpg", frame);


		// video.write(frame);
		// cap >> frame;
    



    return 0;
}