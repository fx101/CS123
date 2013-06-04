#ifndef NN_H_
#define NN_H_

//Network topology and training parameters must be decreasing and specified at compile time!
//This allows for compatibility with < SM 2.0 devices

#define IN 8 //# of input neurons
#define HN 4 //# of hidden neurons
#define ON 1 //# of output neurons
#define LAYERS 3 //# of layers in the network
#define N 16384 //# of samples (Make it a power of 2)
#define LR 0.3 //Learning Rate
#define MOM 0.75 //Momentum

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

__device__ void initActMat(float * ins , float * actMatrix);

__device__ void sliceData(float* glob_data , float* dev_data, bool input);

__global__ void kernBackProp(float* ins, float* outs, float* weights, float* grossUpdates, float* prevNetUp, float* outerrors);

#endif /* NN_H_ */
