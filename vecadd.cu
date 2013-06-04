/*
 * Vector Addition Kernel
 * From NVIDIA CUDA by Example
 */

__global__ void add(int *a, int *b, int *c)
{
	int tID = blockIdx.x;
	if (tID < N)
	{
		c[tID] = a[tID] + b[tID];
	}
}




