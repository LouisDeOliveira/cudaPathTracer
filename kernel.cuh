#pragma once	

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> // printf
#include <iostream> // std::cerr, std::cout
#include <fstream> // std::ifstream
#include <vector> // std::vector
#include <string> // std::string
#include <sstream> // std::istringstream

// CUDA UTILS AND KERNEL WRAPPERS
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define PI 3.14159265359f

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

__global__ void debugKernel();

__global__ void uvKernel(float* framebuffer, int width, int height, float time);

__global__ void renderKernel(float* framebuffer, int width, int height, float time, float3 cameraPos);


void debugKernelWrapper();

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time);
void renderKernelWrapper(uint8_t* framebuffer, int width, int height, float time, float3 cameraPos);


// PATH TRACING UTILS


/*
Struct to represent a ray of light.
*/
struct Ray {
	float3 origin;
	float3 direction;

	__device__ Ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	__device__ float3 at(const float t);
};

struct Sphere
{
	float3 center;
	float radius;

	__host__ __device__ Sphere(float3 center, float radius) : center(center), radius(radius) {}
};

bool __device__ intersectSphere(const Ray& ray, const Sphere& sphere, float& t);


// MESH UTILS

/*
 A struct to represent a triangle Mesh
 vertices: an array of vertices
 faces: an array of faces, each face is a triplet of indices into the vertices array
 vertexNormals: an array of vertex normals, not all meshes have vertex normals
 vertexUVs: an array of vertex UVs, not all meshes have vertex UVs
 AABB: an array of two float3s representing the axis aligned bounding box of the mesh
*/
struct Mesh {
	std::vector<float3> vertices;
	std::vector<float3> faces;
	std::vector<float3> vertexNormals;
	std::vector<float2> vertexUVs;
	float3 AABB[2];
};

void loadObj(const char* filename, Mesh& mesh, float scalefactor=1.0f);


/*
Offsets the position of all the vertices in the mesh by the given vector.
*/
void offsetMesh(Mesh& mesh, float3 offset);

/*
Scales the size of the mesh in each direction by the given vector.
*/
void scaleMesh(Mesh& mesh, float3 scale);

/*
Scales the size of the mesh in every direction by the given factor.
*/
void scaleMesh(Mesh& mesh, float scale);

void computeAABB(Mesh& mesh);

// MATH UTILS

//float3
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator-(const float3& a) {
	return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(const float a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator/(const float3& a, const float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator/(const float a, const float3& b) {
	return make_float3(a / b.x, a / b.y, a / b.z);
}

inline __host__ __device__ float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float length(const float3& a) {
	return sqrtf(dot(a, a));
}

inline __host__ __device__ float3 normalize(const float3& a) {
	return a * (1.0f / length(a));
}

inline __host__ __device__ float3 clamp(const float3& a, float min, float max) {
	return make_float3(fminf(fmaxf(a.x, min), max), fminf(fmaxf(a.y, min), max), fminf(fmaxf(a.z, min), max));
}

inline __host__ __device__ float3 lerp(const float3& a, const float3& b, float t) {
	return a + (b - a) * t;
}

inline __host__ __device__ float3 fminf(const float3& a, const float3& b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

