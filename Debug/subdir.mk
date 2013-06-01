################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../array_reduction.cu \
../main.cu \
../nn.cu 

CU_DEPS += \
./array_reduction.d \
./main.d \
./nn.d 

OBJS += \
./array_reduction.o \
./main.o \
./nn.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I"/home/aluque/cuda-workspace/CS123" -G -g -O0 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=sm_20 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -G -I"/home/aluque/cuda-workspace/CS123" -O0 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


