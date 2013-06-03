/*
 * nn.h
 *
 *  Created on: May 28, 2013
 *      Author: aluque
 */

#ifndef NN_H_
#define NN_H_

//Macros are used for compatibility with CUDA API

//Network topology must be decreasing!
#define IN 8 //# of input neurons
#define HN 4 //# of hidden neurons
#define ON 1 //# of output neurons
#define LAYERS 3 //# of layers in the network
#define N 16384 //# of samples (Make it a power of 2)

__device__ float sigmoid(float *x);

__device__ float DSigmoid(float *x);

__device__ float logit(float *x);

__device__ void initActMat(float * ins , float * actMatrix);

__device__ void sliceData(float* glob_data , float* dev_data);

__global__ void kernBackProp(float* ins, float* outs, float* weights, float* updates);

#endif /* NN_H_ */
