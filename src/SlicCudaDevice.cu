#include "SlicCudaDevice.h"
#include <stdio.h>

#define DIV_UP(x, y ) (x + y - 1) / y


__global__ void kRgb2CIELab(const cudaTextureObject_t texFrameBGRA, cudaSurfaceObject_t surfFrameLab, int width, int height) {

	int px = blockIdx.x*blockDim.x + threadIdx.x;
	int py = blockIdx.y*blockDim.y + threadIdx.y;

	if (px<width && py<height) {
		uchar4 nPixel = tex2D<uchar4>(texFrameBGRA, px, py);//inputImg[offset];

		float _b = (float)nPixel.x / 255.0;
		float _g = (float)nPixel.y / 255.0;
		float _r = (float)nPixel.z / 255.0;

		float x = _r * 0.412453 + _g * 0.357580 + _b * 0.180423;
		float y = _r * 0.212671 + _g * 0.715160 + _b * 0.072169;
		float z = _r * 0.019334 + _g * 0.119193 + _b * 0.950227;

		x /= 0.950456;
		float y3 = exp(log(y) / 3.0);
		z /= 1.088754;

		float l, a, b;

		x = x > 0.008856 ? exp(log(x) / 3.0) : (7.787 * x + 0.13793);
		y = y > 0.008856 ? y3 : 7.787 * y + 0.13793;
		z = z > 0.008856 ? z /= exp(log(z) / 3.0) : (7.787 * z + 0.13793);

		l = y > 0.008856 ? (116.0 * y3 - 16.0) : 903.3 * y;
		a = (x - y) * 500.0;
		b = (y - z) * 200.0;

		float4 fPixel;
		fPixel.x = l;
		fPixel.y = a;
		fPixel.z = b;
		fPixel.w = 0;

		// fPixel.x = (float)nPixel.x;
		// fPixel.y = (float)nPixel.y;
		// fPixel.z = (float)nPixel.z;
		// fPixel.w = (float)nPixel.w;

		surf2Dwrite(fPixel, surfFrameLab, px * 16, py);
	}
}

__global__ void kInitClusters(const cudaSurfaceObject_t surfFrameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol) {
	int centroidIdx = blockIdx.x*blockDim.x + threadIdx.x;
	int nSpx = nSpxPerCol*nSpxPerRow;

	if (centroidIdx<nSpx){
		int wSpx = width / nSpxPerRow;
		int hSpx = height / nSpxPerCol;

		int i = centroidIdx / nSpxPerRow;
		int j = centroidIdx%nSpxPerRow;

		int x = j*wSpx + wSpx / 2;
		int y = i*hSpx + hSpx / 2;

		float4 color;
		surf2Dread(&color, surfFrameLab, x * sizeof(float4), y);
		clusters[centroidIdx] = color.x;
		clusters[centroidIdx + 1 * nSpx] = color.y;
		clusters[centroidIdx + 2 * nSpx] = color.z;
		clusters[centroidIdx + 3 * nSpx] = x;
		clusters[centroidIdx + 4 * nSpx] = y;
	}
}


__global__ void kAssignment_stencil(const cudaSurfaceObject_t surfFrameLab, 
	const float* clusters,
	const int width, 
	const int height, 
	const int wSpx, 
	const int hSpx, 
	const float wc2, 
	cudaSurfaceObject_t surfLabels,
	float* accAtt_g){

	// gather NNEIGH surrounding clusters
	const int NNEIGH = 3;

	int nClustPerRow = width / wSpx;
	int nClustPerCol = height / hSpx;

	int nbSpx = nClustPerRow * nClustPerCol;

	// Find nearest neighbour
	float areaSpx = wSpx*hSpx;
	int px = blockIdx.x*blockDim.x + threadIdx.x;


	int py_num = DIV_UP(height,blockDim.y);
	int py_end = py_num*(threadIdx.y+1) > height? height:py_num*(threadIdx.y+1);
	if (px<width){
		for (int py=py_num*threadIdx.y; py<py_end; py++){
			float distanceMin = 9999999;
			float labelMin = -1;
			float distTmp = distanceMin;

			float4 color;
			surf2Dread(&color, surfFrameLab, px * sizeof(float4), py);
			float3 px_Lab = make_float3(color.x, color.y, color.z);
			float2 px_xy = make_float2(px, py);

			int spx_coor_x = px/wSpx;
			int spx_coor_y = py/hSpx;

			int offset_y[9] = {-1,-1,-1, 0,0,0, 1,1,1};
			int offset_x[9] = {-1, 0, 1,-1,0,1,-1,0,1};

			#pragma unroll
			for (int n=0; n<9;n++){
				int j = offset_x[n];
				int i = offset_y[n];

				int spx_neigh_coor_x = spx_coor_x + j;
				int spx_neigh_coor_y = spx_coor_y + i;
				int spx_idx = spx_neigh_coor_y*nClustPerRow + spx_neigh_coor_x;

				if (spx_neigh_coor_x>=0 && spx_neigh_coor_x<nClustPerRow &&
					spx_neigh_coor_y>=0 && spx_neigh_coor_y<nClustPerCol){

					float2 cluster_xy = make_float2(clusters[spx_idx+3*nbSpx], clusters[spx_idx+4*nbSpx]);
					float3 cluster_Lab = make_float3(clusters[spx_idx], clusters[spx_idx+nbSpx], clusters[spx_idx+2*nbSpx]);
					// float2 cluster_xy = make_float2(0,0);
					// float3 cluster_Lab = make_float3(0,0);


					float2 xy_diff = px_xy - cluster_xy;
					float3 lab_diff = px_Lab - cluster_Lab;
					float ds2 = xy_diff.x*xy_diff.x + xy_diff.y*xy_diff.y;
					float dc2 = lab_diff.x*lab_diff.x + lab_diff.y*lab_diff.y + lab_diff.z*lab_diff.z;
					distTmp = sqrtf(dc2 + ds2 / areaSpx*wc2);

			
					// if (px==0 && py==0){
						// printf("nbSpx:%d, wSpx:%d, hSpx:%d, spx_idx:%d| (px %d, py %d) - (nn_coorx %f, nn_coory %f) = %f, %f\n",
						// 	nbSpx,wSpx,hSpx,spx_idx,
						// 	px,py,clust_x,clust_y,distTmp, distanceMin);
						// printf("%f %f\n", distTmp, distanceMin);
						// printf("dist = %f %f %f %f %f\n", px_c_xy.x,px_c_xy.y,px_c_Lab.x,px_c_Lab.y,px_c_Lab.z);
						// printf("labxy c = %f,%f,%f,%f,%f\n",clust_l,clust_a,clust_b,clust_x,clust_y);
						// printf("labxy p = %f,%f,%f,%f,%f\n",color.x,color.y,color.z,(float)px,(float)py);
					//	}

					if (distTmp < distanceMin){
						distanceMin = distTmp;
						labelMin = spx_idx;
						// if (px==0 && py == 0){
						// 	printf("labelmin %d \n", labelMin);
						// }
					}
					
				}
			}
		
			surf2Dwrite(labelMin, surfLabels, px * sizeof(float), py);
			
			int iLabelMin = int(labelMin);
			atomicAdd(&accAtt_g[iLabelMin            ], px_Lab.x);
			atomicAdd(&accAtt_g[iLabelMin +     nbSpx], px_Lab.y);
			atomicAdd(&accAtt_g[iLabelMin + 2 * nbSpx], px_Lab.z);
			atomicAdd(&accAtt_g[iLabelMin + 3 * nbSpx], px);
			atomicAdd(&accAtt_g[iLabelMin + 4 * nbSpx], py);
			atomicAdd(&accAtt_g[iLabelMin + 5 * nbSpx], 1); //counter*/
		}
	}
		
}


__global__ void kUpdate(int nbSpx, float* clusters, float* accAtt_g)
{
	int cluster_idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (cluster_idx<nbSpx){
		int nbSpx2 = nbSpx * 2;
		int nbSpx3 = nbSpx * 3;
		int nbSpx4 = nbSpx * 4;
		int nbSpx5 = nbSpx * 5;
		int counter = accAtt_g[cluster_idx + nbSpx5];
		if (counter != 0){
			clusters[cluster_idx] = accAtt_g[cluster_idx] / counter;
			clusters[cluster_idx + nbSpx] = accAtt_g[cluster_idx + nbSpx] / counter;
			clusters[cluster_idx + nbSpx2] = accAtt_g[cluster_idx + nbSpx2] / counter;
			clusters[cluster_idx + nbSpx3] = accAtt_g[cluster_idx + nbSpx3] / counter;
			clusters[cluster_idx + nbSpx4] = accAtt_g[cluster_idx + nbSpx4] / counter;

			//reset accumulator
			accAtt_g[cluster_idx] = 0;
			accAtt_g[cluster_idx + nbSpx] = 0;
			accAtt_g[cluster_idx + nbSpx2] = 0;
			accAtt_g[cluster_idx + nbSpx3] = 0;
			accAtt_g[cluster_idx + nbSpx4] = 0;
			accAtt_g[cluster_idx + nbSpx5] = 0;
		}
	}
}