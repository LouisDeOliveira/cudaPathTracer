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

	int xStride = gridDim.x * blockDim.x;
	int yStride = gridDim.y * blockDim.y;

	for (int i = y; i < height; i += yStride) {
		for (int j = x; j < width; j += xStride) {
			int pixelIndex = i * width * 4 + j * 4;

			intFrameBuffer[pixelIndex] = (uint8_t)(floatFrameBuffer[pixelIndex] * 255);
			intFrameBuffer[pixelIndex + 1] = (uint8_t)(floatFrameBuffer[pixelIndex + 1] * 255);
			intFrameBuffer[pixelIndex + 2] = (uint8_t)(floatFrameBuffer[pixelIndex + 2] * 255);
			intFrameBuffer[pixelIndex + 3] = (uint8_t)(floatFrameBuffer[pixelIndex + 3] * 255);
		}
	}
}	


__global__ void debugKernel() {
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void uvKernel(float* framebuffer, int width, int height, float time) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int xStride = gridDim.x * blockDim.x;
	int yStride = gridDim.y * blockDim.y;

	for (int i = y; i < height; i += yStride) {
		for (int j = x; j < width; j += xStride) {
			int pixelIndex = i * width * 4 + j * 4;

			framebuffer[pixelIndex] = (float)j / width;
			framebuffer[pixelIndex + 1] = (float)i / height;
			framebuffer[pixelIndex + 2] = 0.5f + 0.5 * cosf(time);
			framebuffer[pixelIndex + 3] = 1.0f;
		}
	}
}

void debugKernelWrapper() {
	debugKernel <<<2, 2 >> > ();
	cudaDeviceSynchronize();
}

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time) {
	int xBlocks = 64;
	int yBlocks = 64;
	int xThreads = 32;
	int yThreads = 32;
	
	dim3 blockSize(xBlocks, yBlocks);
	dim3 gridSize(xThreads, yThreads);

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

void loadObj(const char* filename, Mesh& mesh, float scalefactor)
{
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Could not open file " << filename << std::endl;
		exit(1);
	}

	std::cout << "Loading file " << filename << std::endl;

	std::string line;
	while (std::getline(file, line))
	{
		//std::cout << line << std::endl;
		
		std::istringstream iss(line);
		std::string prefix;

		iss >> prefix;

		if (prefix == "v") {
			float x, y, z;
			iss >> x >> y >> z;
			mesh.vertices.push_back(make_float3(x * scalefactor, y * scalefactor, z * scalefactor));
		}
		else if (prefix == "f") {
			int v1, v2, v3;
			iss >> v1 >> v2 >> v3;
			mesh.faces.push_back(make_float3(v1 - 1, v2 - 1, v3 - 1));
		}
		else if (prefix == "vn") {
			float x, y, z;
			iss >> x >> y >> z;
			mesh.vertexNormals.push_back(make_float3(x, y, z));
		}
		else if (prefix == "vt") {
			float u, v;
			iss >> u >> v;
			mesh.vertexUVs.push_back(make_float2(u, v));
		}	
	}

	computeAABB(mesh);

	std::cout << "Loaded " << mesh.vertices.size() << " vertices and " << mesh.faces.size() << " faces" << std::endl;

}

void offsetMesh(Mesh& mesh, float3 offset) {
	for (int i = 0; i < mesh.vertices.size(); i++) {
		mesh.vertices[i] = mesh.vertices[i] + offset;
	}
}

void scaleMesh(Mesh& mesh, float3 scale) {
	for (int i = 0; i < mesh.vertices.size(); i++) {
		mesh.vertices[i] = mesh.vertices[i] * scale;
	}
}

void scaleMesh(Mesh& mesh, float scale) {
	for (int i = 0; i < mesh.vertices.size(); i++) {
		mesh.vertices[i] = mesh.vertices[i] * scale;
	}
}

void computeAABB(Mesh& mesh) {
	float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (int i = 0; i < mesh.vertices.size(); i++) {
		min = fminf(min, mesh.vertices[i]);
		max = fmaxf(max, mesh.vertices[i]);
	}

	mesh.AABB[0] = min;
	mesh.AABB[1] = max;
}