/*
 * Kernels For Training ANN
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <nn.h>

__device__ float sigmoid(float *x)
{
	return (1.0/(1.0 + __expf(-*x)));
}

__device__ void initActMat(float * ins , float * actMatrix)
{
	for(int i=0 ; i < IN ; i++)
	{
		actMatrix[i] = ins[threadIdx.x*IN + i];
	}
}

__device__ void sliceData(float* glob_data , float* dev_data)
{
	for(int i=0 ; i < IN ; i++)
		{
			dev_data[i] = glob_data[blockIdx.x*IN + i];
		}
}

__global__ void actNodeCol(float* ins, float* outs, float* weights, float* errors)
{
	//Setup per-block flattened matrices
	__shared__ float inputs[IN]; //on-chip subsection of inputs
	__shared__ float activations[IN*LAYERS];
	__shared__ float dev_weights[IN*IN*(LAYERS-1)];
	*dev_weights = *weights; //on-chip copy of weights

	sliceData(ins, inputs);
	initActMat(inputs,activations);

	//Initialize Inputs
	for(int i = 0; i < IN ; i++)
	{
		activations[IN+i] = sigmoid(&activations[i]);
	}

	//Weighted Sum to Hidden Neuron
	__syncthreads();
	float hnSum = 0.0;
	for(int i = 0; i < IN ; i++)
	{
		hnSum += dev_weights[IN*threadIdx.x + i] * activations[IN+i];
	}
	//Store Output from Hidden Neuron
	activations[2*IN + threadIdx.x] = sigmoid(&hnSum);

	//Weighted Sum to Output Neuron
	__syncthreads();
	if(threadIdx.x < ON)
	{
		float onSum = 0.0;
		for(int i = 0; i < HN ; i++)
		{
			onSum += dev_weights[IN*threadIdx.x + i] * activations[2*IN+i];
		}
		//Store Output from Output Neuron
		activations[3*IN + threadIdx.x] = sigmoid(&onSum);
		//Calculate Squared Error (Assumes errors initialized to 0)
		for(int i = 0; i < ON ; i++)
		{
			errors[blockIdx.x] += __powf(outs[i]-activations[3*IN + i] , 2.0);
		}
	}
}
