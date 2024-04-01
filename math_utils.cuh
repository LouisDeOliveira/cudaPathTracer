#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

