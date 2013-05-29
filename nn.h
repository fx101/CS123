/*
 * nn.h
 *
 *  Created on: May 28, 2013
 *      Author: aluque
 */

#ifndef NN_H_
#define NN_H_

//Macros are used for compatibility with CUDA API
#define IN 8 //# of input neurons
#define HN 4 //# of hidden neurons
#define ON 1 //# of output neurons
#define LAYERS 3 //# of layers in the network

__device__ float sigmoid(float *x);

__device__ void initActMat(float * ins , float * actMatrix);

__device__ void sliceData(float* glob_data , float* dev_data);

__global__ void actNodeCol(float* ins, float* outs, float* weights, float* errors);

#endif /* NN_H_ */
