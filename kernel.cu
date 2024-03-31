#include "kernel.cuh"

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
 {
	 if (result) {
		 std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			 file << ":" << line << " '" << func << "' \n";
		 // Make sure we call CUDA Device Reset before exiting
		 cudaDeviceReset();
		 exit(99);
	 }
}

__global__ void floatToUint8(float* floatFrameBuffer, uint8_t* intFrameBuffer, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width * 4 + x * 4;

	intFrameBuffer[pixelIndex] = (uint8_t)(floatFrameBuffer[pixelIndex] * 255);
	intFrameBuffer[pixelIndex + 1] = (uint8_t)(floatFrameBuffer[pixelIndex + 1] * 255);
	intFrameBuffer[pixelIndex + 2] = (uint8_t)(floatFrameBuffer[pixelIndex + 2] * 255);
	intFrameBuffer[pixelIndex + 3] = (uint8_t)(floatFrameBuffer[pixelIndex + 3] * 255);
}	


__global__ void debugKernel() {
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void uvKernel(float* framebuffer, int width, int height, float time) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width*4 + x*4;

	framebuffer[pixelIndex] = (float)x/width;
	framebuffer[pixelIndex + 1] = (float)y / height;
	framebuffer[pixelIndex + 2] = 0.5f + 0.5*cosf(time);
	framebuffer[pixelIndex + 3] = 1.0f;
}

void debugKernelWrapper() {
	debugKernel <<<2, 2 >> > ();
	cudaDeviceSynchronize();
}

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time) {
	dim3 blockSize(64, 64);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	float* cudaframebuffer;
	uint8_t* cudaIntFrameBuffer;
	checkCudaErrors(cudaMalloc((void**)&cudaframebuffer, sizeof(float) * width * height * 4));
	checkCudaErrors(cudaMalloc((void**)&cudaIntFrameBuffer, sizeof(uint8_t) * width * height * 4));
	 

	uvKernel <<< blockSize, gridSize >>> (cudaframebuffer, width, height, time);
	cudaDeviceSynchronize();
	floatToUint8 <<< blockSize, gridSize >>> (cudaframebuffer, cudaIntFrameBuffer, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(framebuffer, cudaIntFrameBuffer, sizeof(uint8_t) * width * height * 4, cudaMemcpyDeviceToHost);
	cudaFree(cudaframebuffer);
	cudaFree(cudaIntFrameBuffer);
	//printf("Kernel finished\n");
}