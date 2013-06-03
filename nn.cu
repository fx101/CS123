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

__device__ float DSigmoid(float *x)
{
	return sigmoid(x)*(1-sigmoid(x));
}

__device__ float logit(float *x)
{
	return __logf(*x) - __logf(1.0-*x);
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

__global__ void kernBackProp(float* ins, float* outs, float* weights, float* updates)
{
	/*
	 * Weights are flattened in the following form:
	 * Weight array is subdivided into (Layers-1) partitions of size IN^2 that correspond to layers
	 * Each of these partitions is subdivided into HN partitions of size IN
	 *
	 */
	__shared__ float inputs[IN]; //on-chip subsection of inputs
	__shared__ float activations[IN*LAYERS];
	__shared__ float dev_weights[(IN+1)*HN*(LAYERS-1)];
	__shared__ float partSums[IN*LAYERS];

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
		hnSum += dev_weights[(IN+1)*threadIdx.x + i] * activations[IN+i];
	}
	hnSum += dev_weights[(IN+1) * threadIdx.x + IN]; //hidden bias

	//Store Output from Hidden Neuron
	activations[2*IN + threadIdx.x] = sigmoid(&hnSum);

	//Weighted Sum to Output Neuron
	__syncthreads();
	if(threadIdx.x < ON)
	{
		float onSum = 0.0;
		for(int i = 0; i < HN ; i++)
		{
			onSum += dev_weights[HN*(IN+1) + (IN+1)*threadIdx.x + i] * activations[2*IN+i];
		}
		onSum += dev_weights[HN*(IN+1) + (IN+1)*threadIdx + IN]; //output bias
		//Output Neuron Activations
		activations[3*IN + threadIdx.x] = sigmoid(&onSum);

		//Sq Error
		for(int i = 0; i < ON ; i++)
		{
			errors[blockIdx.x] += (outs[i]-activations[3*IN + i])*(outs[i]-activations[3*IN + i]);
		}
		errors[blockIdx.x] *= 0.5;
	}

	//Backpropagate



}

