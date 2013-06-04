/*
 *Alfredo Luque
 *CUDA 3-layer MLP
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <nn.h>
#include <array_reduction.h>
#include <csv_v2.h>

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

void sumArray(float* errIn, float* partsum , float* error)
{
	//sum errors to blockdim numbers
	sumreduce<<<N/blockSize,blockSize>>>(errIn,partsum,N); //reduces into redBlocks floats
	sumreduce<<<1,blockSize>>>(partsum,error,N); //reduces to final sum
}

void sumVec(float* a , float* b, float* c) //a+b=c
{
	// Fill Arrays
	for (int i = 0; i < N; i++)
		cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
		add<<<N,1>>>(dev_a, dev_b, dev_c);
		cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);
}

float* loadInputs(float* dev_array, bool inputfile)
{
  if(inputfile)
  {
	  io::CSVReader<3> in("inputs.csv"); //just rename the file inputs.csv
  }
  else
  {
	  io::CSVReader<3> in("outputs.csv"); //just rename the file outputs.csv
  }
  in.read_header(io::ignore_extra_column, "Weather", "OnTime", "Delay/Out");
  std::string vendor; int size; double speed;
  while(in.read_row(vendor, size, speed)){
	  for(int i = 0 ; i < IN)
	  {
		  cudaMemcpy(row[i], dev_in , N*sizeof(float), cudaMemcpyHostToDevice);
	  }
  }
}

int main(void)
{
	//load training data

	//Finalize block dimensions
	//Initialize Weights
	unsigned int nW = (unsigned int)((IN+1)*HN*(LAYERS-1)); //One bias per hidden and output layer
	float* wSeeds;
	wSeeds = initW(nW);
	float* dev_w;
	cudaMalloc((void**)&dev_w,(sizeof(float)*nW));
	cudaMemcpy(wSeeds , dev_w , sizeof(float)*nW , cudaMemcpyHostToDevice);

	//Generate Device Error Array
	float* dev_errIn;
	float* dev_errPartSum;
	float* dev_error;
	cudaMalloc((void**)&dev_errIn,(sizeof(float)*N));
	cudaMalloc((void**)&dev_errPartSum,(sizeof(float)*N));
	cudaMalloc((void**)&dev_error,sizeof(float));
	//Generate Device Gross Update Array
	float* dev_grossUp;
	cudaMalloc((void**)&dev_grossUp,N*sizeof(float)*nW);
	//Generate Prev Update Arrays (Momentum)
	float* dev_prevUp;
	cudaMalloc((void**)&dev_prevUp,sizeof(float)*nW);

	float* dev_tdi
	cudaMalloc((void**)&dev_prevUp,sizeof(float)*IN*N);
	float* dev_tdo;
	cudaMalloc((void**)&dev_prevUp,sizeof(float)*ON*N); //usually just size N*sizeof(float) since I have one output neuron
	loadinputs(dev_tdi,true);
	loadinputs(dev_tdo,false);
	//Iterate Backpropagation!
	for(int i = 0 ; i < EPOCHS ; i++)
	{
		kernBackProp<<<N,HN>>>(dev_tdi, dev_tdo, dev_w ,dev_grossUp , dev_prevUp dev_errIn);
		sumArray(dev_errIn, dev_error); //we can output this somewhere to create convergence charts
		for(int j = 0 ; j < N ; j++)
		{
			sumVec(&grossUp[i] , &grossUp[i+1] , &dev_weights); //compute net changes
		}
		cudaMemcpy(dev_weights, wei)
	}
	cudaMemcpy(dev_weights, weights, (sizeof(float)*nW) , cudaMemcpyDeviceToHost ); //retrieve weights (that's what we're after!).
	return 0;
}
