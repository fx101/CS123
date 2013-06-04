/*
 * Kernels For Training ANN
 * Optimized for One Hidden Layer
 * IMPORTANT: Arbitrary Layer Connectivity Requires Nesting in additional loop. Use NVCC loop unrolling flags to guarantee good performance.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <nn.h>



__device__ void initActMat(float * ins , float * actMatrix)
{
	for(int i=0 ; i < IN ; i++)
	{
		actMatrix[i] = ins[threadIdx.x*IN + i];
	}
}

__device__ void sliceData(float* glob_data , float* dev_data, bool input)
{
	if(input == true) //Slice Inputs
	{
	for(int i=0 ; i < IN ; i++)
		{
			dev_data[i] = glob_data[blockIdx.x*IN + i];
		}

	}
	else //Slice Outputs
	{
		for(int i=0 ; i < ON ; i++)
		{
			dev_data[i] = glob_data[blockIdx.x*ON + i];
		}

	}
}

__global__ void kernBackProp(float* ins, float* outs, float* weights, float* grossUpdates, float* prevNetUp, float* outerrors)
{
	/*
	 * Weights are flattened in the following form:
	 * Weight array is subdivided into (Layers-1) partitions of size HN*(IN+1) that correspond to layers
	 * Each of these partitions is subdivided into HN partitions of size IN+1 (up to IN inputs and 1 bias)
	 *
	 */
	__shared__ float inputs[IN]; //on-chip subsection of inputs
	__shared__ float outputs[ON];
	__shared__ float activations[IN*LAYERS];
	__shared__ float dev_weights[(IN+1)*HN*(LAYERS-1)];
	__shared__ float partSums[IN*LAYERS];
	__shared__ float deltas[(IN+1)*HN*(LAYERS-1)];
	__shared__ float outdeltas[ON];

	*dev_weights = *weights; //on-chip copy of weights

	sliceData(ins, inputs,true);
	sliceData(outs, outputs,false);
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
		//Output Deltas
		outdeltas[threadIdx.x] = outputs[threadIdx.x]-activations[3*IN + threadIdx.x];

		//Weight Changes Hidden:Output
		for(int i = 0 ; i < (HN+1) ; i++)
		{
			deltas[HN*(IN+1) + (IN+1)*threadIdx.x + i] = dev_weights[HN*(IN+1) + (IN+1)*threadIdx.x + i]*outdeltas[threadIdx.x]*DSigmoid(&activations[3*IN + threadIdx.x]);

			//No getting around the insane indices. Apologies to the brave soul who reads this.
			//Basically... change = -learningrate*delta*activation + momentum*previousweightchange
			grossUpdates[BlockIdx.x*((IN+1)*HN*(LAYERS-1))+(HN*(IN+1) + (IN+1)*threadIdx.x + i)]=(-1.0)*(LR*activations[HN*(IN+1) + (IN+1)*threadIdx.x + i]*deltas[HN*(IN+1) + (IN+1)*threadIdx.x + i] +(prevNetUp[HN*(IN+1) + (IN+1)*threadIdx.x + i]*MOM));
		}
	}
	__syncthreads();
	//Weight Changes From Hidden to Input Layer
	for(int i = 0 ; i < IN+1 ; i++)
	{
		for(int i = 0; i < HN ; i++) //sum over all hidden neurons
		{
			deltas[(IN+1)*threadIdx.x + i] = dev_weights[(IN+1)*threadIdx.x + i]*deltas[HN*(IN+1) + (IN+1)*threadIdx.x + i]*DSigmoid(&activations[2*IN + threadIdx.x]);
			grossUpdates[BlockIdx.x*((IN+1)*HN*(LAYERS-1))+((IN+1)*threadIdx.x + i)] = (-1.0)*(LR*activations[2*IN + threadIdx.x]*deltas[(IN+1)*threadIdx.x + i])+(MOM*prevNetUp[(IN+1)*threadIdx.x + i]);
		}
	}

	//We store output sum squared at end of kernel to mitigate warp divergence
	if(threadIdx.x == 0)
	{
		for(int i = 0 ; i < ON ; i++)
		{
			outErrors[BlockIdx.x] = (outputs[i]-activations[3*IN + i])*(outputs[i]-activations[3*IN + i]);
		}
	}
}

