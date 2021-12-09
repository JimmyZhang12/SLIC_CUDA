/*
Superpixel oversegmentation
GPU implementation of the algorithm SLIC of
Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]

Library required :
Opencv 3.0 min
CUDA arch>=3.0

Author : Derue Fran�ois-Xavier
francois.xavier.derue@gmail.com


*/

#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }




class Point2 {
public:
    __host__ __device__ Point2() {}
    __host__ __device__ Point2(int _x, int _y)
    : x(_x), y(_y) {}
    
    int x, y;
    int error = INT_MAX;

    __host__ __device__ bool isInvalid() {
            bool b1 = (x == -1);
        bool b2 = (y == -1);
        return b1 && b2;
    }
    
};
struct Triangle
{
    Point2 points[3];
    int num_points = 0;
    bool removed = false;
    __host__ __device__
    Triangle(){}

    __host__ __device__
    Triangle(Point2 a, Point2 b, Point2 c) {
        points[0] = a;
        points[1] = b;
        points[2] = c;
    }

    __host__ __device__
    Point2 center() {
        int x = (points[0].x + points[1].x + points[2].x) / 3;
        int y = (points[0].y + points[1].y + points[2].y) / 3;
        return Point2(x, y);
        // return (points[0] + points[1] + points[2]) / 3;
    }


    __host__ __device__
    int min_x() {
		int tmp = points[0].x<points[1].x?points[0].x:points[1].x;
		return (tmp<points[2].x)?tmp:points[2].x;
    }
    __host__ __device__
    int max_x() {
		int tmp = points[0].x>points[1].x?points[0].x:points[1].x;
		return (tmp>points[2].x)?tmp:points[2].x;
	}
    __host__ __device__
    int min_y() {
		int tmp = points[0].y<points[1].y?points[0].y:points[1].y;
		return (tmp<points[2].y)?tmp:points[2].y;
    }
    __host__ __device__
    int max_y() {
		int tmp = points[0].y>points[1].y?points[0].y:points[1].y;
		return (tmp>points[2].y)?tmp:points[2].y;
    }


    __host__ __device__
	float area(Point2 p1, Point2 p2, Point2 p3){
		return abs((p1.x*(p2.y-p3.y) + p2.x*(p3.y-p1.y)+ p3.x*(p1.y-p2.y))/2.0);
	}	
	//https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/
    __host__ __device__
	bool isInside(Point2 point){
		/* Calculate area of triangle ABC */
		float A = area(points[0],points[1],points[3]);
		/* Calculate area of triangle PBC */ 
		float A1 = area(point,points[1],points[3]);
		/* Calculate area of triangle PAC */ 
		float A2 = area(points[0],point,points[3]);
		/* Calculate area of triangle PAB */  
		float A3 = area(points[0],points[1],point);
		/* Check if sum of A1, A2 and A3 is same as A */
		return (A == A1 + A2 + A3);
	}
};

class SlicCuda {
public:
	enum InitType{
		SLIC_SIZE, // initialize with a size of spx
		SLIC_NSPX // initialize with a number of spx
	};
private:
	const int m_deviceId = 0;

	cudaDeviceProp m_deviceProp;

	int m_nbPx;
	int m_nbSpx;
	int m_SpxDiam;
	int m_SpxWidth, m_SpxHeight, m_SpxArea;
	int m_FrameWidth, m_FrameHeight;
	float m_wc;
	int m_nbIteration;
	InitType m_InitType;
	

	//cpu buffer
	float* h_fClusters;
	float* h_fLabels;
	float assignment_time_count=0.0;
	
	// gpu variable
	float* d_fClusters;
	float* d_fLabels;
	float* d_fAccAtt;

	//cudaArray
	cudaArray* cuArrayFrameBGRA;
	cudaArray* cuArrayFrameLab;
	cudaArray* cuArrayLabels;

	// Texture and surface Object
	cudaTextureObject_t oTexFrameBGRA;
	cudaSurfaceObject_t oSurfFrameLab;
	cudaSurfaceObject_t oSurfLabels;

	//========= methods ===========

	void initGpuBuffers(); 
	void uploadFrame(const cv::Mat& frameBGR); 
	void gpuRGBA2Lab();

	/*
	Initialize centroids uniformly on a grid with a step of diamSpx
	*/
	void gpuInitClusters();
	void downloadLabels();

	/*
	Assign the closest centroid to each pixel
	*/
	void assignment(); 

	/*
	Update the clusters' centroids with the belonging pixels
	*/
	void update(); 

public:
	SlicCuda();
	SlicCuda(const cv::Mat& frame0, const int diamSpxOrNbSpx = 15, const InitType initType = SLIC_SIZE, const float wc = 35, const int nbIteration = 5);
	~SlicCuda();

	/*
	Set up the parameters and initalize all gpu buffer for faster video segmentation.
	*/
	void initialize(const cv::Mat& frame0, const int diamSpxOrNbSpx = 15, const InitType initType = SLIC_SIZE, const float wc = 35, const int nbIteration = 5);
	
	/*
	Segment a frame in superpixel
	*/
	void segment(const cv::Mat& frame);
	cv::Mat getLabels(){ return cv::Mat(m_FrameHeight, m_FrameWidth, CV_32F, h_fLabels); }

	/*
	Discard orphan clusters (optional)
	*/
	int enforceConnectivity();

	// cpu draw
	static void displayBound(cv::Mat& image, const float* labels, const cv::Scalar colour);

    static void displayPoint(cv::Mat& image, const float* labels, const cv::Scalar colour);
    static void displayPoint1(cv::Mat& image, const float* labels, const cv::Scalar colour);

	static void CalcLocalError(cv::Mat& image, Point2 *h_deviceOnwers, Triangle *triangles, int num_triangles);
	static std::vector<Triangle> getAdjacentTriangles(Point2 point, Triangle *triangles, int num_triangles);

	static std::string type2str(int type);


};

static inline int iDivUp(int a, int b){ return (a%b == 0) ? a / b : a / b + 1; }

/*
Find best width and height from a given diameter to best fit the image size given by imWidth and imHeigh
*/
static void getSpxSizeFromDiam(const int imWidth, const int imHeight, const int diamSpx, int* spxWidth, int* spxHeight);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
