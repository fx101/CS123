/*
 *Alfredo Luque
 *CUDA 3-layer MLP
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <nn.h>

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



int main(void)
{
	//Initialize Weights
	unsigned int nW = (unsigned int)(pow(IN,2.0)*2.0); //max dim for square weight matrix
	float* wSeeds;
	wSeeds = initW(nW);
	float* dev_wSeeds;
	cudaMalloc((void**)&dev_wSeeds,(sizeof(float)*nW));
	cudaMemcpy(wSeeds , dev_wSeeds , sizeof(float)*nW , cudaMemcpyDeviceToHost);

	//load data
	return 0;
}
