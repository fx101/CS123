################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu \
../nn.cu 

CU_DEPS += \
./main.d \
./nn.d 

OBJS += \
./main.o \
./nn.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/aluque/cuda-workspace/NeuralNet" -G -g -O0 -gencode arch=compute_10,code=sm_10 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I"/home/aluque/cuda-workspace/NeuralNet" -O0 -g -gencode arch=compute_10,code=compute_10 -gencode arch=compute_10,code=sm_10  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


