/*
 * Sums elements of Array
 * Must be operated in multiple kernel calls:
 * Generally, 2-step reduction.
 *
 * Will only work on compute capability >= 2.0
 *
 * Adapted from Nvidia Developer Presentation
 */
#include<array_reduction.h>
__global__ void sumreduce(float *g_idata, float *g_odata, unsigned int n)
	/*
	 * g_idata pointer to inputs on global device memory
	 * g_idata pointer to outputs on global device memory
	 * n number of floats to add
	 */
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32)
	{
		if (blockSize >= 64)
		{
			sdata[tid] += sdata[tid + 32];
		}
		if (blockSize >= 32)
		{
			sdata[tid] += sdata[tid + 16];
		}
		if (blockSize >= 16)
		{
			sdata[tid] += sdata[tid + 8];
		}
		if (blockSize >= 8)
		{
			sdata[tid] += sdata[tid + 4];
		}
		if (blockSize >= 4)
		{
			sdata[tid] += sdata[tid + 2];
		}
		if (blockSize >= 2)
		{
			sdata[tid] += sdata[tid + 1];
		}
	}
	if (tid == 0) //write to global mem
	{
		g_odata[blockIdx.x] = sdata[0];
	}
	return;
}
