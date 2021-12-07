#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "SlicCudaHost.h"

using namespace std;
using namespace cv;

int main() {
 
	// VideoCapture cap("/home/jimmy/ece508/data/waterfall.avi");


	// Parameters
	int diamSpx = 15;
	int wc = 35;
	int nIteration = 5;
	SlicCuda::InitType initType = SlicCuda::SLIC_SIZE;

	//start segmentation
	Mat frame;
	frame = imread("/home/shun/Desktop/1080.jpg", IMREAD_COLOR);

	// cap >> frame;

	SlicCuda oSlicCuda;
	oSlicCuda.initialize(frame, diamSpx, initType, wc, nIteration);

	// int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	// int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	// int endFrame = cap.get(CAP_PROP_FRAME_COUNT);
    // double fps = cap.get(CAP_PROP_FPS);

    // VideoWriter video("/home/jimmy/ece508/segs/waterfall.avi", 
	// 	VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width,frame_height));

	Mat labels;

	// int i = 0;
    // for(i = 0; i<10; i++){
		frame = imread("/home/shun/Desktop/1080.jpg", IMREAD_COLOR);

		auto t0 = std::chrono::high_resolution_clock::now();
		oSlicCuda.segment(frame);

		auto t1 = std::chrono::high_resolution_clock::now();
		double time = std::chrono::duration<double>(t1-t0).count() ;
		// cout << "Frame " << frame.size() <<" "<< i+1 << "/" << endFrame << ", Time: "<< time <<"s"<<endl;

		oSlicCuda.enforceConnectivity();


		labels = oSlicCuda.getLabels();
        auto data = labels.data;
        // for (int i=0; i<128; ++i) {
        //     printf("%d\n", data[i]);
        // }
		// SlicCuda::displayBound(frame, (float*)labels.data, Scalar(0, 0, 0));
		SlicCuda::displayPoint1(frame, (float*)labels.data, Scalar(0, 0, 0));
		imwrite("/home/shun/Desktop/segment.jpg", frame);


		// video.write(frame);
		// cap >> frame;
    // }

	// cap.release();
	// video.release();

    return 0;
}