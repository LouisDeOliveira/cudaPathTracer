#pragma once	

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

__global__ void debugKernel();

__global__ void uvKernel(float* framebuffer, int width, int height, float time);

void debugKernelWrapper();

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time);