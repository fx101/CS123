/*
 * Parallel Array Addition
 *
 * Float adaptation of "Optimizing Parallel Reduction in CUDA" by Mark Harris
 * Blocksize must be power of 2
 */

#ifndef ARRAY_REDUCTION_H_
#define ARRAY_REDUCTION_H_

#define blockSize 256 //Use powers of 2 > 128

__global__ void sumreduce(float *g_idata, float *g_odata, unsigned int n);

#endif /* ARRAY_REDUCTION_H_ */
