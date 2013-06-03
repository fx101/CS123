/*
 *Alfredo Luque
 *CUDA 3-layer MLP
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <nn.h>
#include <array_reduction.h>

using namespace std;

float* initW(unsigned int nW)
{
	float * weights;
	weights = new float[nW];
	srand(time(0));
	for(int i = 0; i < nW ; i++)
	{
		weights[i] =  ((float)((rand() % 1001)-1000))/1000.0;
	}
	return weights;
}

void sumArray(unsigned int numBlocks)
{
	//sum errors to blockdim numbers
	sumreduce<<redBlocks,blockSize>>(dev_errIn,dev_PartSum); //reduces into redBlocks floats
	sumreduce<<1,redBlocks>>(dev_PartSum,dev_error); //reduces to final sum
}


int main(void)
{
	//load training data

	//Finalize block dimensions
	unsigned int redBlocks = N/blockSize;
	//Initialize Weights
	unsigned int nW = (unsigned int)(IN*HN*(LAYERS-1));
	float* wSeeds;
	wSeeds = initW(nW);
	float* dev_w;
	cudaMalloc((void**)&dev_w,(sizeof(float)*nW));
	cudaMemcpy(wSeeds , dev_w , sizeof(float)*nW , cudaMemcpyDeviceToHost);

	//Generate Device Error Array
	float* dev_errIn;
	float* dev_errPartSum;
	float* dev_error;
	cudaMalloc((void**)&dev_errIn,(sizeof(float)*N));
	cudaMalloc((void**)&dev_errPartSum,(sizeof(float)*N));
	cudaMalloc((void**)&dev_error,sizeof(float));
	//propagate network
	actNodeCol<<N,HN>>(dev_tdi, dev_tdo, dev_w , dev_errIn);

	return 0;
}
