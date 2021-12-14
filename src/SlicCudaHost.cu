#include "SlicCudaHost.h"
#include "SlicCudaDevice.h"
#include <chrono>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <bits/stdc++.h>
#include <thrust/sort.h>
#include "delaunator.h"

using namespace std;
using namespace cv;

int *global_device_error;
int *global_point_mask;

#define NUM_POINTS_REMOVAL_PER_ITER 32

SlicCuda::SlicCuda(){
	int nbGpu = 0;
	gpuErrchk(cudaGetDeviceCount(&nbGpu));
	cout << "Detected " << nbGpu << " cuda capable gpu" << endl;
	gpuErrchk(cudaSetDevice(m_deviceId));
	gpuErrchk(cudaGetDeviceProperties(&m_deviceProp, m_deviceId));
	if (m_deviceProp.major < 3){
		cerr << "compute capability found = " << m_deviceProp.major << ", compute capability >= 3 required !" << endl;
		exit(EXIT_FAILURE);
	}
}

SlicCuda::~SlicCuda(){
	delete[] h_fClusters;
	delete[] h_fLabels;
	gpuErrchk(cudaFree(d_fClusters));
	gpuErrchk(cudaFree(d_fAccAtt));
	gpuErrchk(cudaFreeArray(cuArrayFrameBGRA));
	gpuErrchk(cudaFreeArray(cuArrayFrameLab));
	gpuErrchk(cudaFreeArray(cuArrayLabels));
}

void SlicCuda::initialize(const cv::Mat& frame0, const int diamSpxOrNbSpx , const InitType initType, const float wc , const int nbIteration ) {
	m_nbIteration = nbIteration;
	m_FrameWidth = frame0.cols;
	m_FrameHeight = frame0.rows;
	m_nbPx = m_FrameWidth*m_FrameHeight;
	m_InitType = initType;
	m_wc = wc;
	if (m_InitType == SLIC_NSPX){
		m_SpxDiam = diamSpxOrNbSpx; 
		m_SpxDiam = (int)sqrt(m_nbPx / (float)diamSpxOrNbSpx);
	}
	else m_SpxDiam = diamSpxOrNbSpx;
	
	getSpxSizeFromDiam(m_FrameWidth, m_FrameHeight, m_SpxDiam, &m_SpxWidth, &m_SpxHeight); // determine w and h of Spx based on diamSpx
	m_SpxArea = m_SpxWidth*m_SpxHeight;
	CV_Assert(m_nbPx%m_SpxArea == 0); 
	m_nbSpx = m_nbPx / m_SpxArea; 

	h_fClusters = new float[m_nbSpx * 5]; // m_nbSpx * [L,a,b,x,y]
	h_fLabels = new float[m_nbPx];

	initGpuBuffers();
}

void SlicCuda::segment(const Mat& frameBGR) {
	uploadFrame(frameBGR);
	gpuRGBA2Lab();
	gpuInitClusters();

	for (int i = 0; i<m_nbIteration; i++) {
		// auto t0 = std::chrono::high_resolution_clock::now();
		assignment();
		update();
		// auto t1 = std::chrono::high_resolution_clock::now();
		// double time = std::chrono::duration<double>(t1-t0).count() ;
		// cout<<std::fixed<<i<< ": Total segment Time: "<< time <<"s"<<endl;
	}
	downloadLabels();
}

void SlicCuda::initGpuBuffers() {
	//allocate buffers on gpu

	cudaChannelFormatDesc channelDescrBGRA = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	gpuErrchk(cudaMallocArray(&cuArrayFrameBGRA, &channelDescrBGRA, m_FrameWidth, m_FrameHeight));

	cudaChannelFormatDesc channelDescrLab = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayFrameLab, &channelDescrLab, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

	cudaChannelFormatDesc channelDescrLabels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayLabels, &channelDescrLabels, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

	// Specify texture frameBGRA object parameters
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArrayFrameBGRA;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;
	gpuErrchk(cudaCreateTextureObject(&oTexFrameBGRA, &resDesc, &texDesc, NULL));

	// surface frameLab
	cudaResourceDesc rescDescFrameLab;
	memset(&rescDescFrameLab, 0, sizeof(rescDescFrameLab));
	rescDescFrameLab.resType = cudaResourceTypeArray;

	rescDescFrameLab.res.array.array = cuArrayFrameLab;
	gpuErrchk(cudaCreateSurfaceObject(&oSurfFrameLab, &rescDescFrameLab));

	// surface labels
	cudaResourceDesc resDescLabels;
	memset(&resDescLabels, 0, sizeof(resDescLabels));
	resDescLabels.resType = cudaResourceTypeArray;

	resDescLabels.res.array.array = cuArrayLabels;
	gpuErrchk(cudaCreateSurfaceObject(&oSurfLabels, &resDescLabels));
	
	// buffers clusters , accAtt
	gpuErrchk(cudaMalloc((void**)&d_fClusters, m_nbSpx*sizeof(float) * 5)); // 5-D centroid
	gpuErrchk(cudaMalloc((void**)&d_fAccAtt, m_nbSpx*sizeof(float) * 6)); // 5-D centroid acc + 1 counter
	cudaMemset(d_fAccAtt, 0, m_nbSpx*sizeof(float) * 6);//initialize accAtt to 0
	
}


void SlicCuda::uploadFrame(const Mat& frameBGR) { 
	cv::Mat frameBGRA;
	cv::cvtColor(frameBGR, frameBGRA, COLOR_BGR2BGRA);

	CV_Assert(frameBGRA.type() == CV_8UC4);
	CV_Assert(frameBGRA.isContinuous());
	gpuErrchk(cudaMemcpyToArray(cuArrayFrameBGRA, 0, 0, (uchar*)frameBGRA.data, m_nbPx* sizeof(uchar4), cudaMemcpyHostToDevice));
	

	/*uchar* dst = new uchar[4 * m_nbPx];
	cudaMemcpyFromArray(dst, cuArrayFrameBGRA, 0, 0, m_nbPx*sizeof(uchar4), cudaMemcpyDeviceToHost);
	Mat matDst(m_FrameHeight, m_FrameWidth, CV_8UC4, dst);
	cout << matDst << endl;*/
}

void SlicCuda::gpuRGBA2Lab() {
	const int blockW = 16; 
	const int blockH = blockW;
	CV_Assert(blockW*blockH <= m_deviceProp.maxThreadsPerBlock);
	dim3 threadsPerBlock(blockW, blockH);
	dim3 numBlocks(iDivUp(m_FrameWidth, blockW), iDivUp(m_FrameHeight, blockH));

	kRgb2CIELab << <numBlocks, threadsPerBlock >> >(oTexFrameBGRA, oSurfFrameLab, m_FrameWidth, m_FrameHeight);

	/*float* dst = new float[4 * m_nbPx];
	cudaMemcpyFromArray(dst, cuArrayFrameLab, 0, 0, m_nbPx*sizeof(float4), cudaMemcpyDeviceToHost);
	Mat matDst(m_FrameHeight, m_FrameWidth, CV_32FC4, dst);
	cout << matDst << endl;*/
}



void SlicCuda::gpuInitClusters() {
	int blockW = 16;
	dim3 threadsPerBlock(blockW);
	dim3 numBlocks(iDivUp(m_nbSpx, blockW));

	kInitClusters << <numBlocks, threadsPerBlock >> >(oSurfFrameLab,
		d_fClusters,
		m_FrameWidth, 
		m_FrameHeight,
		m_FrameWidth / m_SpxWidth, 
		m_FrameHeight / m_SpxHeight);

	/*float* fTmp = new float[m_nbSpx * 5];
	cudaMemcpy(fTmp, d_fClusters, m_nbSpx * 5 * sizeof(float), cudaMemcpyDeviceToHost);
	Mat matTmp(1, m_nbSpx*5, CV_32F, fTmp);
	cout << matTmp << endl;*/
}

void SlicCuda::assignment(){
	int PENCILS_PER_BLOCK = 32;
	int VERTICAL_SPLIT = 8;

	int numBlock = iDivUp(m_FrameWidth,PENCILS_PER_BLOCK);
	dim3 blockPerGrid(numBlock);
	dim3 threadPerBlock(PENCILS_PER_BLOCK,VERTICAL_SPLIT);
	// printf("Assignment GRID (%d,%d), BLOCK (%d,%d)\n",
	// 	 blockPerGrid.x, blockPerGrid.y,
	// 	 threadPerBlock.x, threadPerBlock.y);


	float wc2 = m_wc * m_wc;

	auto t0 = std::chrono::high_resolution_clock::now();
	kAssignment_stencil << < blockPerGrid, threadPerBlock >> >(oSurfFrameLab,
		d_fClusters,
		m_FrameWidth, 
		m_FrameHeight, 
		m_SpxWidth,
		m_SpxHeight,
		wc2, 
		oSurfLabels, 
		d_fAccAtt);
	cudaDeviceSynchronize();

	auto t1 = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double>(t1-t0).count();
	
	// cout<<std::fixed<<"\tAssignment Time: "<< time <<"s"<<endl;
	// assignment_time_count += time;
}

void SlicCuda::update(){
	dim3 threadsPerBlock(m_deviceProp.maxThreadsPerBlock);
	dim3 numBlocks(iDivUp(m_nbSpx, m_deviceProp.maxThreadsPerBlock));
	kUpdate << <numBlocks, threadsPerBlock >> >(m_nbSpx, d_fClusters, d_fAccAtt);
	cudaDeviceSynchronize();

}

void SlicCuda::downloadLabels(){
	cudaMemcpyFromArray(h_fLabels, cuArrayLabels, 0, 0, m_nbPx* sizeof(float), cudaMemcpyDeviceToHost);
}

int SlicCuda::enforceConnectivity() {
	int label = 0, adjlabel = 0;
	int lims = (m_FrameWidth * m_FrameHeight) / (m_nbSpx);
	lims = lims >> 2;

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	vector<vector<int> >newLabels;
	for (int i = 0; i < m_FrameHeight; i++) {
		vector<int> nv(m_FrameWidth, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_FrameHeight; i++) {
		for (int j = 0; j < m_FrameWidth; j++){
			if (newLabels[i][j] == -1){
				vector<cv::Point> elements;
				elements.push_back(cv::Point(j, i));
				for (int k = 0; k < 4; k++){
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
						if (newLabels[y][x] >= 0){
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++){
					for (int k = 0; k < 4; k++){
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
							if (newLabels[y][x] == -1 && h_fLabels[i*m_FrameWidth + j] == h_fLabels[y*m_FrameWidth + x]){
								elements.push_back(cv::Point(x, y));
								newLabels[y][x] = label;//m_labels[i][j];
								count += 1;
							}
						}
					}
				}
				if (count <= lims) {
					for (int c = 0; c < count; c++) {
						newLabels[elements[c].y][elements[c].x] = adjlabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}
	int nbSpxNoOrphan = label; // new number of spx
	for (int i = 0; i < newLabels.size(); i++)
		for (int j = 0; j < newLabels[i].size(); j++)
			h_fLabels[i*m_FrameWidth + j] = (float)newLabels[i][j];

	return nbSpxNoOrphan;
}

int NextPower2_CPU(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__device__ __inline__ int Index(Point2 p, int col)
{
    return p.y * col + p.x;
}

__device__ int dist(const Point2 &a, const Point2 &b)
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx*dx + dy*dy;
}

__device__ __inline__ bool InBound(Point2 p, int row, int col)
{
    return (0 <= p.x && p.x < col && 0<=p.y && p.y < row);
}


__host__ __device__ Point2 operator + (const Point2 &a, const Point2 &b)
{
    return Point2(a.x + b.x, a.y + b.y);
}

__host__ __device__ Point2 operator * (const Point2 &a, int b)
{
    return Point2(a.x * b, a.y * b);
}

__host__ __device__ Point2 operator / (const Point2 &a, int b)
{
    return Point2(a.x / b, a.y / b);
}

__host__ __device__ bool operator == (const Point2 &a, const Point2 &b)
{
    return a.x == b.x && a.y == b.y;
}

__device__ void count_colors(Point2 now_point, Point2 device_owner[], Point2 colors[4], int &numColors, int cols)
{
    Point2 neighbor_dir[] = {Point2(1, 0), Point2(0, 1), Point2(1, 1)};

    colors[0] = device_owner[Index(now_point, cols)];
    numColors = 1;
    for(Point2 now_dir: neighbor_dir)
    {
        Point2 next_point = now_point + now_dir;
        Point2 newColor = device_owner[Index(next_point, cols)];
        bool exist = false;
        for (int i = 0; i < numColors; i++)
        {
            if (newColor == colors[i])
            {
                exist = true;
                break;
            }
        }
        if (!exist)
        {
            colors[numColors] = newColor;
            numColors ++;
        }
    }
}

__global__ void count_triangle_kernel(Point2* device_owner, int rows, int cols, int* triangle_count)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= cols-1 || r >= rows-1)
        return;

    Point2 now_point(c, r);

    Point2 colors[4] = {Point2(-1, -1), Point2(-1, -1), Point2(-1, -1), Point2(-1, -1)};
    int numColors;
    count_colors(now_point, device_owner, colors, numColors, cols);

    if(numColors == 3)
        triangle_count[Index(now_point, cols)] = 1;
    else if(numColors == 4)
        triangle_count[Index(now_point, cols)] = 2;
    else
        triangle_count[Index(now_point, cols)] = 0;
}

__global__ void triangle_kernel(Point2* device_owner, Triangle* device_triangles, int rows, int cols, int* device_sum_triangles)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // __shared__ int block_triangle_index;
    // if (c == 0 && r == 0) {
    //     block_triangle_index = 0;
    // }
    // __syncthreads();

    if (c >= cols-1 || r >= rows-1)
        return;

    Point2 now_point(c, r);

    Point2 colors[4] = {Point2(-1, -1), Point2(-1, -1), Point2(-1, -1), Point2(-1, -1)};
    int numColors;
    count_colors(now_point, device_owner, colors, numColors, cols);

    int index = Index(now_point, cols);
    int prev_triangle_cnt = (index == 0)? 0: device_sum_triangles[index-1];

    if(numColors == 3)
    {
        Triangle triangle;
        for(int i=0; i<3; i++)
        {
            triangle.points[i] = colors[i];
        }
        // device_triangles[atomicAdd(&block_triangle_index, 1)] = triangle;
        device_triangles[prev_triangle_cnt] = triangle;
    }
    else if(numColors == 4)
    {
        Triangle triangle1(device_owner[Index(now_point, cols)],
                           device_owner[Index(now_point + Point2(1, 0), cols)],
                           device_owner[Index(now_point + Point2(0, 1), cols)]);

        Triangle triangle2(device_owner[Index(now_point + Point2(1, 0), cols)],
                           device_owner[Index(now_point + Point2(0, 1), cols)],
                           device_owner[Index(now_point + Point2(1, 1), cols)]);

        // device_triangles[atomicAdd(&block_triangle_index, 1)] = triangle1;
        // device_triangles[atomicAdd(&block_triangle_index, 1)] = triangle2;
        device_triangles[prev_triangle_cnt] = triangle1;
        device_triangles[prev_triangle_cnt + 1] = triangle2;
    }
    return;
}


__device__ __inline__ int signGPU(Point2 p1, Point2 p2, Point2 p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ bool PointInTriangleGPU(Point2 pt, Point2 v1, Point2 v2, Point2 v3)
{
    int d1, d2, d3;
    bool has_neg, has_pos;

    d1 = signGPU(pt, v1, v2);
    d2 = signGPU(pt, v2, v3);
    d3 = signGPU(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}


__device__ int binary_search_right(int col_l, int col_r, int row, const Point2& p1, const Point2& p2, const Point2& p3) {
    while (col_l < col_r) {
        int col_m = col_l + (col_r - col_l) / 2;
        Point2 p(col_m, row);
        if (PointInTriangleGPU(p, p1, p2, p3)) {
            col_l = col_m + 1;
        } else {
            col_r = col_m;
        }
    }
    return col_l;
}


__global__ void draw_triangle_kernel1(Triangle* device_triangles, int num_triangles, uint8_t* device_img, uint8_t* device_tri_img, int rows, int cols)
{
    int triIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (triIdx >= num_triangles)
        return;

    Triangle tri = device_triangles[triIdx];
    if (tri.removed) return;
    int minX = min(tri.points[0].x, tri.points[1].x);
    minX = min(minX, tri.points[2].x);
    int minY = min(tri.points[0].y, tri.points[1].y);
    minY = min(minY, tri.points[2].y);    
    int maxX = max(tri.points[0].x, tri.points[1].x);
    maxX = max(maxX, tri.points[2].x);
    int maxY = max(tri.points[0].y, tri.points[1].y);
    maxY = max(maxY, tri.points[2].y);  
    Point2 p = tri.center();
    int imgIdx = (p.y * cols + p.x) * 3;

    uint8_t pr = device_img[imgIdx];
    uint8_t pg = device_img[imgIdx + 1];
    uint8_t pb = device_img[imgIdx + 2];

    for (int r = minY; r < maxY; ++r)
    {
        for (int c = minX; c < maxX; ++c)
        {
            if (r>=0 && r<rows && c>=0 && c<cols) {
                Point2 pt(c, r);
                if (PointInTriangleGPU(pt, tri.points[0], tri.points[1], tri.points[2]))    
                { 
                    device_triangles[triIdx].num_points ++;
                    int col_l = c;
                    // int col_r = binary_search_right(col_l, maxX, r, tri.points[0], tri.points[1], tri.points[2]);
                    // for (int c=col_l; c<col_r; ++c) {
                        int triImgIdx = (r * cols + c) * 3;
                        device_tri_img[triImgIdx] = pr; 
                        device_tri_img[triImgIdx + 1] = pg; 
                        device_tri_img[triImgIdx + 2] = pb; 
                    // }    
                    // break;
                    // int triImgIdx = (r * cols + c) * 3;
                    // device_tri_img[triImgIdx] = pr; 
                    // device_tri_img[triImgIdx + 1] = pg; 
                    // device_tri_img[triImgIdx + 2] = pb; 
                }   
            }
            
        }       
        

    }
}


bool isNoCornerArround( std::vector<std::vector<unsigned char>> & corner, const int& height, const int& width, const int& x, const int& y )
{
	if(x==0 || x==width-1 || y==0 || y==height-1) return true;

	if( corner[y+1][x+1] == 255 ) return false;
	if( corner[y+1][x] == 255 ) return false;
	if( corner[y+1][x-1] == 255 ) return false;
	if( corner[y][x-1] == 255 ) return false;
	if( corner[y-1][x-1] == 255 ) return false;
	if( corner[y-1][x] == 255 ) return false;
	if( corner[y-1][x+1] == 255 ) return false;
	if( corner[y][x+1] == 255 ) return false;
	
	return true;
}

std::vector<int> get_triangles(Triangle *triangles, int num_triangles, int i, int j) {
    std::vector<int> loc;

    for (int t=0; t<num_triangles; ++t) {
        if (triangles[t].num_points > 0 && !triangles->removed) {
            for (Point2 p: triangles[t].points) {
                if (p.y == i && p.x == j) {
                    loc.push_back(t);
                    break;
                }
            }
        }
    }
    return loc;
}

int calculate_error(cv::Mat& image, const int height, const int width, Triangle *triangles, int num_triangles, int i, int j) {

    std::vector<int> loc = get_triangles(triangles, num_triangles, i, j);

    int error = 0;
    Vec3b curr = image.at<cv::Vec3b>(i, j);
    int curr_r = curr[0], curr_g = curr[1], curr_b = curr[2];
    for (int ti: loc) {
        auto t = triangles[ti];
        Point2 p = t.center();

        Vec3b color = image.at<cv::Vec3b>(p.y, p.x);
        int r = color[0], g = color[1], b = color[2];
        error += pow(curr_r - r, 2) * t.num_points + pow(curr_g - g, 2) * t.num_points + pow(curr_b - b, 2) * t.num_points;
    }
    return error;
}

__global__ void check_device_owner(Point2* device_owner, int stepsize, int rows, int cols) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < rows * cols) {
        Point2 p = device_owner[tid];
        if (p.isInvalid()) {
            return;
        }
        if (p.x < 0 || p.y < 0) {
            printf(">>>>>>>>>>>>>>>>invalid device owner: %d\n", tid);
        }

    }
}

#define MAX_TRIANGLE_PER_PIXEL 8

__global__ void group_triangle_kernel(Triangle *d_triangles, int num_triangles, int *pixel_triangle_cnt, int *pixel_triangle_indices, int rows, int cols) {
    int triIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (triIdx >= num_triangles)
        return;
    
    Triangle t = d_triangles[triIdx];
    for (Point2 p: t.points) {
        if (p.isInvalid()) continue;
        int row = p.y, col = p.x;
        int ptr = row * cols + col;
        int idx = atomicAdd(&pixel_triangle_cnt[ptr], 1);
        pixel_triangle_indices[MAX_TRIANGLE_PER_PIXEL * ptr + idx] = triIdx;
    }
}

__global__ void calculate_error_gpu(uint8_t *image, Point2 *device_owners, const int rows, const int cols, Triangle* triangles, int *pixel_triangle_cnt, int *pixel_triangle_indice, int *error) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < rows * cols) {
        Point2 p = device_owners[tid];
        if (p.isInvalid()) {
            error[tid] = 2147483647;
            return;
        }

        int img_idx = tid * 3;
        int r = image[img_idx], g = image[img_idx+1], b = image[img_idx+2];

        int err = 0;
        int n_triangles = pixel_triangle_cnt[tid];
        for (int i=0; i<n_triangles; ++i) {
            int triIdx = pixel_triangle_indice[tid * 8 + i];
            Triangle t = triangles[triIdx];
            Point2 p = t.center();
            
            int col_idx = (p.y * cols + p.x) * 3;
            int pr = image[col_idx], pg = image[col_idx + 1], pb = image[col_idx + 2];
            err += (pr-r) * (pr-r) * t.num_points + (pg-g) * (pg-g) * t.num_points + (pb-b) * (pb-b) * t.num_points;
        }
        error[tid] = err;
    }
}



__global__ void voronoi_kernel_fixed(Point2* device_owner, Point2* double_buf, int stepsize, int rows, int cols, bool db)
{   

    int y_offset[9] = {-1,-1,-1, 0,0,0, 1,1,1};
    int x_offset[9] = {-1, 0, 1,-1,0,1,-1,0,1};

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;


    if (c >= cols || r >= rows)
        return;

    Point2 now_point(c, r);
    Point2 now_owner = db?double_buf[Index(now_point, cols)]:device_owner[Index(now_point, cols)];

    int min_dist = INT_MAX;
    Point2 closest_point;
    Point2 closest_owner;

    for (int i=0; i<=9; i++){
        Point2 looking = now_point + Point2(x_offset[i],y_offset[i]) * stepsize;

        if(!InBound(looking, rows, cols))
            continue;
        Point2 looking_owner = db?double_buf[Index(looking, cols)]:device_owner[Index(looking, cols)];
        if (looking_owner.isInvalid())
            continue;

        int cand_dist = dist(looking_owner, now_point);
        if((cand_dist < min_dist)){
            closest_point = looking;
            min_dist = cand_dist;
            closest_owner = looking_owner;
        }
    }
    if (min_dist != INT_MAX){
        if (db)
            device_owner[Index(now_point, cols)] = closest_owner;
        else
            double_buf[Index(now_point, cols)] = closest_owner;

    }

}



__host__ void launch_voronoi_kernel(Point2* d_ownerMap, int rows, int cols, dim3 gridDim, dim3 blockDim){    
    int start_stepsize = NextPower2_CPU(min(rows, cols)) / 2;
    Point2 *double_buf;

    cudaMalloc(&double_buf, sizeof(Point2) * rows * cols);
    bool db = false;
    // cudaMemcpy(double_buf, d_ownerMap, sizeof(Point2) * rows * cols, cudaMemcpyDeviceToDevice);

    for(int stepsize = start_stepsize; stepsize>=1; stepsize /= 2)
    {
        voronoi_kernel_fixed<<<gridDim, blockDim>>>(d_ownerMap, double_buf, stepsize, rows, cols, db);
        gpuErrchk(cudaDeviceSynchronize());       
        db = !db;
    }
    cudaFree(double_buf);   
    if (db)
        cudaMemcpy(d_ownerMap, double_buf, sizeof(Point2) * rows * cols, cudaMemcpyDeviceToDevice);

    check_device_owner<<<(rows * cols - 1) / 128 + 1, 128>>>(d_ownerMap, 1, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());
}



void all_t(int rows, int cols, Point2* d_ownerMap, Point2* h_deviceOnwers, int *d_triangle_sum, uint8_t* d_tri_img, uint8_t* d_img, int &num_triangles, cv::Mat& image, Triangle* h_triangles) {
    cudaMemcpy(d_ownerMap, h_deviceOnwers, sizeof(Point2) * rows * cols, cudaMemcpyHostToDevice);
    unsigned int n = 32;
    dim3 blockDim(n, n);
    dim3 gridDim((cols + n - 1) / n, (rows + n - 1) / n);

    launch_voronoi_kernel(d_ownerMap, rows, cols, gridDim, blockDim);


    int *d_triangle_cnts;
    cudaMalloc(&d_triangle_cnts, sizeof(int) * rows * cols);

    count_triangle_kernel<<<gridDim, blockDim>>>(d_ownerMap, rows, cols, d_triangle_cnts);
    gpuErrchk(cudaDeviceSynchronize());

    // int* device_sum_triangles;
    thrust::inclusive_scan(thrust::device, (int*)d_triangle_cnts, (int*)d_triangle_cnts + rows*cols, (int*)d_triangle_sum);

    cudaMemcpy(&num_triangles, &((int*)d_triangle_sum)[rows*cols-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_triangle_cnts);

    // num_triangles = 6000;

    Triangle *d_triangles;
    cudaMalloc(&d_triangles, sizeof(Triangle) * (num_triangles));
    triangle_kernel<<<gridDim, blockDim>>>(d_ownerMap, d_triangles, rows, cols, d_triangle_sum);
    gpuErrchk(cudaDeviceSynchronize());


    cudaMemcpy(d_img, image.data, sizeof(uint8_t)*rows * cols*3, cudaMemcpyHostToDevice);

    // cudaMemcpy(num_triangles, &((int*)d_triangle_sum)[rows*cols-1], sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMalloc(d_triangles, sizeof(Triangle) * (*num_triangles));
    // triangle_kernel<<<gridDim, blockDim>>>((Point*)d_owner_map, (Triangle*)(*d_triangles), rows, cols, (int*)d_triangle_sum);
    // gpuErrchk(cudaDeviceSynchronize());
    int threadPerBlock1 = 96;
    int gridDim1 = (num_triangles - 1) / threadPerBlock1 + 1;

    draw_triangle_kernel1<<<gridDim1, threadPerBlock1>>>((Triangle*)d_triangles, num_triangles, (uint8_t*)d_img, (uint8_t*)d_tri_img, rows, cols); 
    gpuErrchk(cudaDeviceSynchronize());  
    cudaMemcpy(h_triangles, d_triangles, sizeof(Triangle) * num_triangles, cudaMemcpyDeviceToHost);

    int *pixel_triangle_cnt;
    cudaMalloc(&pixel_triangle_cnt, sizeof(int) * rows * cols);
    cudaMemset(pixel_triangle_cnt, 0, sizeof(int) * rows * cols);

    int *pixel_triangle_index;
    cudaMalloc(&pixel_triangle_index, sizeof(int) * rows * cols * MAX_TRIANGLE_PER_PIXEL);

    group_triangle_kernel<<<gridDim1, threadPerBlock1>>>(d_triangles, num_triangles, pixel_triangle_cnt, pixel_triangle_index, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());  

    int num_threads = 96;
    int num_blocks = (rows * cols - 1) / num_threads + 1;
    cudaMemcpy(d_ownerMap, h_deviceOnwers, sizeof(Point2) * rows * cols, cudaMemcpyHostToDevice);

    calculate_error_gpu<<<num_blocks, num_threads>>>(d_img, d_ownerMap, rows, cols, d_triangles, pixel_triangle_cnt, pixel_triangle_index, global_device_error);
    gpuErrchk(cudaDeviceSynchronize());  
    
    thrust::sequence(thrust::device, global_point_mask, global_point_mask + rows * cols, 0);
    thrust::sort_by_key(thrust::device, global_device_error, global_device_error + rows * cols, global_point_mask);

    cudaFree(pixel_triangle_cnt);
    cudaFree(pixel_triangle_index);


    cudaFree(d_triangles);
}

void SlicCuda::displayPoint1(cv::Mat& image, const float* labels, const cv::Scalar colour) {
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */

    std::vector<std::vector<unsigned char>> zeros(image.rows, std::vector<unsigned char>(image.cols, 0));
    std::vector<std::vector<unsigned char>> corner(image.rows, std::vector<unsigned char>(image.cols, 0));
    std::vector<std::vector<int>> errors(image.rows, std::vector<int>(image.cols, INT_MAX));

    Point2 *h_deviceOnwers = new Point2[image.rows * image.cols];

	vector<cv::Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}
    int width = image.cols, height = image.rows;

	// Go through all the pixels.
    auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
            if ((j % 100 == 0 && i == 0 ) || (i % 120 == 0 && j == 0)) {
                contours.push_back(cv::Point(j, i));
				istaken[i][j] = true;
                continue;
            }

			int nr_p = 0;
			// Compare the pixel to its 8 neighbours.
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && labels[i*image.cols + j] != labels[y*image.cols + x]) {
						nr_p += 1;
					}
				}
			}
            if (nr_p > 1) {
                istaken[i][j] = true;
                zeros[i][j] = 255;
            }
		}
	}

	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
            if( zeros[i][j]==255 ) {
                std::map<int,int> map;
                int count = 1;
                for(int p=0; p<=1; p++) for(int q=0; q<=1; q++) {
                    int ny = i + p;
                    int nx = j + q;
                    if( nx>=0 && nx<width && ny>=0 && ny<height ) {
                        if( map.find(labels[ny*width+nx])==map.end() ) {
                            map.insert(pair<int,int>(labels[ny*width+nx],count));
                            count++;
                        }
                    }
                }
                if( count > 3) {
                    if( isNoCornerArround(corner,height,width,j, i) ) {
                        corner[i][j] = 255;
                    }
                }
            }

		}
	}

	corner[0][0] = 255;
	corner[height-1][0] = 255;
	corner[0][width-1] = 255;
	corner[height-1][width-1] = 255;
	for(int y=0; y<height; y++) {
		if( zeros[y][0]==255 ) {
			corner[y][0] = 255;
		}
		if( zeros[y][width-1]==255 ) {
			corner[y][width-1] = 255;
		}
	}

	for(int x=0; x<width; x++) {
		if( zeros[0][x]==255 ) {
			corner[0][x] = 255;
		}
		if( zeros[height-1][x]==255 ) {
			corner[height-1][x] = 255;
		}
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double>(t1-t0).count();
	cout<<std::fixed<<"Corner Selection Time: "<< time <<"s"<<endl;

    // printf("Vertices size: %d\n", contours.size());
	// Draw the contour pixels. 
	// for (int i = 0; i < (int)contours.size(); i++) {
	// 	image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b((uchar)colour[0], (uchar)colour[1], (uchar)colour[2]);
	// }
    // return;
    int num_vert = 0;
    for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
            if (corner[i][j] == 255) {
                h_deviceOnwers[i * image.cols + j].x = j;
                h_deviceOnwers[i * image.cols + j].y = i;
                //  = Point2(i, j);
                num_vert ++;
            } else {

                h_deviceOnwers[i * image.cols + j] = Point2(-1, -1);
            }
        }
    }
    // for (int i = 0; i < (int)contours.size(); i++) {
    //     int r = contours[i].y, c = contours[i].x;
    // }
    printf("Num vertices: %d\n", num_vert);
    
    int rows = image.rows, cols = image.cols;

    Point2 *d_ownerMap;
    int *device_sum_triangles, *d_triangle_sum;
    uint8_t *d_img, *d_tri_img;
    cudaMalloc(&d_triangle_sum, sizeof(int) * rows * cols);
    cudaMalloc(&d_ownerMap, sizeof(Point2) * rows * cols);
    cudaMalloc(&device_sum_triangles, sizeof(int) * rows * cols);
    cudaMalloc(&d_img, sizeof(uint8_t) * rows * cols * 3);
    cudaMalloc(&d_tri_img, sizeof(uint8_t) * rows * cols * 3);
    cudaMalloc(&global_device_error, sizeof(int) * rows * cols);
    cudaMalloc(&global_point_mask, sizeof(int) * rows * cols);
    Triangle *h_triangles = new Triangle[80000];
    
    int *host_error = new int[rows * cols];
    int *host_map  = new int[rows * cols];
    int num_triangles;
    all_t(rows, cols, d_ownerMap, h_deviceOnwers, d_triangle_sum, d_tri_img, d_img, num_triangles, image, h_triangles);
    int iter = 0;
    for (int nv=num_vert; nv>1024; nv-=NUM_POINTS_REMOVAL_PER_ITER,iter++)  {
        cudaMemcpy(host_error, global_device_error, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_map, global_point_mask, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

        for (int i=0; i<NUM_POINTS_REMOVAL_PER_ITER; ++i) {
            h_deviceOnwers[host_map[i]].x = -1;
            h_deviceOnwers[host_map[i]].y = -1;
        }

        // int min_err = INT_MAX, min_i = -1;
        // for (int i=0; i<rows * cols; ++i) {
        //     if (h_deviceOnwers[i].isInvalid()) continue;
        //     int err = host_error[i];
        //     if (err < min_err) {
        //         min_err = err;
        //         min_i = i;
        //         // printf("removing point: (%d, %d)\n", min_i, min_j);
        //         // break;
        //     }
        // }
        // if (min_i > -1) {
        //     h_deviceOnwers[min_i].x = -1;
        //     h_deviceOnwers[min_i].y = -1;
        //     all_t(rows, cols, d_ownerMap, h_deviceOnwers, d_triangle_sum, d_tri_img, d_img, num_triangles, image, h_triangles);
        //     // cudaMemcpy(image.data, d_tri_img, sizeof(uint8_t) * rows * cols, cudaMemcpyDeviceToHost);
        //     cudaDeviceSynchronize();
        // } else {
        //     break;
        // }

        all_t(rows, cols, d_ownerMap, h_deviceOnwers, d_triangle_sum, d_tri_img, d_img, num_triangles, image, h_triangles);
        
        // all_t(rows, cols, d_ownerMap, h_deviceOnwers, d_triangle_sum, d_tri_img, d_img, num_triangles, image, h_triangles);
        if (iter % 32 == 0) printf("current nb points: %d\n", nv);
        if (iter % 4 == 0) {
            Mat tmp;
            tmp.create(cv::Size2d(cols, rows), CV_8UC3);
            cudaMemcpy(tmp.data, d_tri_img, sizeof(uint8_t) * rows * cols * 3, cudaMemcpyDeviceToHost);
            imwrite("/home/shun/Desktop/segment.jpg", tmp);
        }
    }


    cudaMemcpy(image.data, d_tri_img, sizeof(uint8_t) * rows * cols * 3, cudaMemcpyDeviceToHost);


    cudaFree(d_img);
    cudaFree(d_tri_img);
    cudaFree(d_triangle_sum);
    cudaFree(device_sum_triangles);
    cudaFree(d_ownerMap);
    cudaFree(global_device_error);

    delete[]h_triangles;
    delete[] host_error;
    delete[] host_map;

    // cudaFree(d_ownerMap);
}

void SlicCuda::displayPoint(cv::Mat& image, const float* labels, const cv::Scalar colour) { }

void SlicCuda::displayBound(cv::Mat& image, const float* labels, const cv::Scalar colour){
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<cv::Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}
	// Go through all the pixels.
	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;
			// Compare the pixel to its 8 neighbours.
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && labels[i*image.cols + j] != labels[y*image.cols + x]) {
						nr_p += 1;
					}
				}
			}
			/* Add the pixel to the contour list if desired. */
			if (nr_p > 1) {
				contours.push_back(cv::Point(j, i));
				istaken[i][j] = true;
			}

		}
	}
	// Draw the contour pixels. 
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b((uchar)colour[0], (uchar)colour[1], (uchar)colour[2]);
	}
}
static void getSpxSizeFromDiam(const int imWidth, const int imHeight, const int diamSpx, int* spxWidth, int* spxHeight){
	int wl1, wl2;
	int hl1, hl2;
	wl1 = wl2 = diamSpx;
	hl1 = hl2 = diamSpx;

	while ((imWidth%wl1) != 0) {
		wl1++;
	}
	while ((imWidth%wl2) != 0) {
		wl2--;
	}
	while ((imHeight%hl1) != 0) {
		hl1++;
	}

	while ((imHeight%hl2) != 0) {
		hl2--;
	}
	*spxWidth = ((diamSpx - wl2) < (wl1 - diamSpx)) ? wl2 : wl1;
	*spxHeight = ((diamSpx - hl2) < (hl1 - diamSpx)) ? hl2 : hl1;
}


std::string SlicCuda::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}