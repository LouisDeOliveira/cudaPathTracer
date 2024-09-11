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

__global__ void SphereKernel(float* framebuffer, int width, int height, float time, float3 cameraPos, float3 cameradir) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int xStride = gridDim.x * blockDim.x;
	int yStride = gridDim.y * blockDim.y;

	Sphere sphere(make_float3(0.5f*cosf(time), 0.5f*sinf(time), -5.0f), 1.0f);


	float aspectRatio = (float)width / height;
	float fov = 30.0f;

	
	Camera camera(cameraPos, make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), 30.0f, width, height);
	camera.setDirection(normalize(cameradir));


	for (int i = y; i < height; i += yStride) {
		for (int j = x; j < width; j += xStride) {
			// Find pixel ray
			float v = (float)(i+0.5f) / height;
			float u = (float)(j+0.5f) / width;

		    //float3 rayDir = make_float3((-2.0f * u + 1.0f) * aspectRatio * tanf(fov / 2.0f), (2.0f * v-1.0f) * tanf(fov / 2.0f), -1.0f); // in cam coords

			// Transform to world coords

			/*Ray ray(camera.position, normalize(rayDir));*/

			//Ray ray(camera.position, normalize(camera.cameraToWorld(rayDir)));
			Ray ray = camera.getRay(u, v);



			int pixelIndex = i * width * 4 + j * 4;

			float3 color;
			HitRecord record;

			if (intersectSphere(ray, sphere, record)) {

				color = 0.5f * make_float3(record.normal.x + 1.0f, record.normal.y + 1.0f, record.normal.z + 1.0f);
			}
			else {
				float a = 0.5f * (ray.direction.y + 1.0f);
				color = (1.0f - a) * make_float3(1.0f, 1.0f, 1.0f) + a * make_float3(0.5f, 0.7f, 1.0f);
			}


			framebuffer[pixelIndex] = color.x;
			framebuffer[pixelIndex + 1] = color.y;
			framebuffer[pixelIndex + 2] = color.z;
			framebuffer[pixelIndex + 3] = 1.0f;

		}
	}

}


void debugKernelWrapper() {
	debugKernel <<<2, 2 >> > ();
	cudaDeviceSynchronize();
}

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time) {
	int xBlocks = 16;
	int yBlocks = 16;
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
}

void SphereKernelWrapper(uint8_t* framebuffer, int width, int height, float time, float3 camerapos, float3 cameradir)
{
	int xBlocks = 16;
	int yBlocks = 16;
	int xThreads = 32;
	int yThreads = 32;

	dim3 blockSize(xBlocks, yBlocks);
	dim3 gridSize(xThreads, yThreads);

	float* cudaframebuffer;
	uint8_t* cudaIntFrameBuffer;
	checkCudaErrors(cudaMalloc((void**)&cudaframebuffer, sizeof(float) * width * height * 4));
	checkCudaErrors(cudaMalloc((void**)&cudaIntFrameBuffer, sizeof(uint8_t) * width * height * 4));


	SphereKernel << < blockSize, gridSize >> > (cudaframebuffer, width, height, time, camerapos, cameradir);
	cudaDeviceSynchronize();
	floatToUint8 << < blockSize, gridSize >> > (cudaframebuffer, cudaIntFrameBuffer, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(framebuffer, cudaIntFrameBuffer, sizeof(uint8_t) * width * height * 4, cudaMemcpyDeviceToHost);
	cudaFree(cudaframebuffer);
	cudaFree(cudaIntFrameBuffer);
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
			mesh.faces.push_back(make_int3(v1 - 1, v2 - 1, v3 - 1));
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
	mesh.AABB[0] = mesh.AABB[0] + offset;
	mesh.AABB[1] = mesh.AABB[1] + offset;
}

void scaleMesh(Mesh& mesh, float3 scale) {
	for (int i = 0; i < mesh.vertices.size(); i++) {
		mesh.vertices[i] = mesh.vertices[i] * scale;
	}
	mesh.AABB[0] = mesh.AABB[0] * scale;
	mesh.AABB[1] = mesh.AABB[1] * scale;
}

void scaleMesh(Mesh& mesh, float scale) {
	for (int i = 0; i < mesh.vertices.size(); i++) {
		mesh.vertices[i] = mesh.vertices[i] * scale;
	}
	mesh.AABB[0] = mesh.AABB[0] * scale;
	mesh.AABB[1] = mesh.AABB[1] * scale;
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

bool __device__ intersectSphere(const Ray& ray, const Sphere& sphere, HitRecord& record) {
	float3 oc = ray.origin - sphere.center;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0f * dot(oc, ray.direction);
	float c = dot(oc, oc) - sphere.radius * sphere.radius;
	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0) {
		return false;
	}
	else {
		float t0 = (-b - sqrt(discriminant)) / (2.0f * a);
		float t1 = (-b + sqrt(discriminant)) / (2.0f * a);

		if (t0 > t1) {
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}

		if (t0 < 0) {
			t0 = t1;
			if (t0 < 0) {
				return false;
			}
		}

		record.t = t0;
		record.point= ray.at(t0);
		record.normal = normalize(record.point - sphere.center);
		return true;
	}
}

bool __device__ intersectTriangle(const Ray& ray, const float3& vertex0, const float3& vertex1, const float3& vertex2, HitRecord& record){
	float3 edge1 = vertex1 - vertex0;
    float3 edge2 = vertex2 - vertex0;

    // Calculate the determinant
    float3 h = cross(ray.direction, edge2);
    float det = dot(edge1, h);

    // If the determinant is near zero, the ray is parallel to the triangle
    if (fabs(det) < 1e-5) {
        return false;
    }

    float invDet = 1.0f / det;

    // Calculate distance from vertex0 to ray origin
    float3 s = ray.origin - vertex0;

    // Calculate u parameter and test bounds
    float u = dot(s, h) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    // Calculate v parameter and test bounds
    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // Calculate t to find out where the intersection point is on the line
    float t = dot(edge2, q) * invDet;

    // If t is greater than EPSILON, ray intersects the triangle
    if (t > 1e-5) {
        record.t = t;
        record.point = ray.origin + ray.direction * t;
        record.normal = cross(edge1, edge2);
        return true;
    }

    // No intersection
    return false;
}





