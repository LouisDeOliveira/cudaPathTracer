#pragma once	

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "math_utils.cuh"
#include <math.h>
#include <float.h>
#include <stdint.h>
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

__global__ void SphereKernel(float* framebuffer, int width, int height, float time, float3 cameraPos, float3 cameradir);


void debugKernelWrapper();

void uvKernelWrapper(uint8_t* framebuffer, int width, int height, float time);
void SphereKernelWrapper(uint8_t* framebuffer, int width, int height, float time, float3 cameraPos, float3 cameraDir);

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

inline void printf(float3 v) {
	printf("%f %f %f\n", v.x, v.y, v.z);
}

//float4
inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

inline __host__ __device__ float4 operator-(const float4& a, const float4& b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ float4 operator-(const float4& a) {
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

inline __host__ __device__ float4 operator*(const float4& a, const float4& b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w*b.w);
}

inline __host__ __device__ float4 operator*(const float4& a, const float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w*b);
}

inline __host__ __device__ float4 operator*(const float a, const float4& b) {
	return make_float4(a * b.x, a * b.y, a * b.z, a*b.w);
}

inline __host__ __device__ float4 operator/(const float4& a, const float4& b) {
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w/b.w);
}

inline __host__ __device__ float4 operator/(const float4& a, const float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w/b);
}

inline __host__ __device__ float4 operator/(const float a, const float4& b) {
	return make_float4(a / b.x, a / b.y, a / b.z, a/b.w);
}

inline __host__ __device__ float dot(const float4& a, const float4& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w*b.w;
}

inline __host__ __device__ float length(const float4& a) {
	return sqrtf(dot(a, a));
}

inline __host__ __device__ float4 normalize(const float4& a) {
	return a * (1.0f / length(a));
}

inline __host__ __device__ float4 clamp(const float4& a, float min, float max) {
	return make_float4(fminf(fmaxf(a.x, min), max), fminf(fmaxf(a.y, min), max), fminf(fmaxf(a.z, min), max), fminf(fmaxf(a.w, min), max));
}

inline __host__ __device__ float4 lerp(const float4& a, const float4& b, float t) {
	return a + (b - a) * t;
}

inline __host__ __device__ float4 fminf(const float4& a, const float4& b) {
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ float4 fmaxf(const float4& a, const float4& b) {
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline void printf(float4 v) {
	printf("%f %f %f %f\n", v.x, v.y, v.z, v.w);
}



// matrices 

struct Matrix3x3 {
	float3 r0;
	float3 r1;
	float3 r2;

	__host__ __device__ Matrix3x3(float3 r0, float3 r1, float3 r2) : r0(r0), r1(r1), r2(r2) {}

	__host__ __device__ static Matrix3x3 fromColumns(float3 c0, float3 c1, float3 c2) {
		return Matrix3x3(make_float3(c0.x, c1.x, c2.x), make_float3(c0.y, c1.y, c2.y), make_float3(c0.z, c1.z, c2.z));
	}

	__host__ __device__ static Matrix3x3 Identity() {
		return Matrix3x3(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
	}
};

inline __host__ __device__ Matrix3x3 operator*(const Matrix3x3& m, const float s) {
	return Matrix3x3(m.r0 * s, m.r1 * s, m.r2 * s);
}

inline __host__ __device__ Matrix3x3 operator*(const float s, const Matrix3x3& m) {
	return Matrix3x3(m.r0 * s, m.r1 * s, m.r2 * s);
}

inline __host__ __device__ Matrix3x3 operator*(const Matrix3x3& a, const Matrix3x3& b) {
	return Matrix3x3(
		make_float3(dot(a.r0, make_float3(b.r0.x, b.r1.x, b.r2.x)), dot(a.r0, make_float3(b.r0.y, b.r1.y, b.r2.y)), dot(a.r0, make_float3(b.r0.z, b.r1.z, b.r2.z))),
		make_float3(dot(a.r1, make_float3(b.r0.x, b.r1.x, b.r2.x)), dot(a.r1, make_float3(b.r0.y, b.r1.y, b.r2.y)), dot(a.r1, make_float3(b.r0.z, b.r1.z, b.r2.z))),
		make_float3(dot(a.r2, make_float3(b.r0.x, b.r1.x, b.r2.x)), dot(a.r2, make_float3(b.r0.y, b.r1.y, b.r2.y)), dot(a.r2, make_float3(b.r0.z, b.r1.z, b.r2.z)))
	);
}

inline __host__ __device__ float3 operator*(const Matrix3x3& m, const float3& v) {
	return make_float3(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

inline __host__ __device__ Matrix3x3 operator+(const Matrix3x3& a, const Matrix3x3& b) {
	return Matrix3x3(a.r0 + b.r0, a.r1 + b.r1, a.r2 + b.r2);
}

inline __host__ __device__ Matrix3x3 operator-(const Matrix3x3& a, const Matrix3x3& b) {
	return Matrix3x3(a.r0 - b.r0, a.r1 - b.r1, a.r2 - b.r2);
}

inline __host__ __device__ Matrix3x3 operator-(const Matrix3x3& a) {
	return Matrix3x3(-a.r0, -a.r1, -a.r2);
}

inline __host__ __device__ float determinant(const Matrix3x3& m) {
	return dot(m.r0, cross(m.r1, m.r2));
}

inline __host__ __device__ Matrix3x3 transpose(const Matrix3x3& m) {
	return Matrix3x3(make_float3(m.r0.x, m.r1.x, m.r2.x), make_float3(m.r0.y, m.r1.y, m.r2.y), make_float3(m.r0.z, m.r1.z, m.r2.z));
}

inline __host__ __device__ Matrix3x3 inverse(const Matrix3x3& m) {
	float det = determinant(m);
	if (det == 0.0f) {
		return Matrix3x3(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f));
	}
	float invDet = 1.0f / det;
	float3 c0 = cross(m.r1, m.r2) * invDet;
	float3 c1 = cross(m.r2, m.r0) * invDet;
	float3 c2 = cross(m.r0, m.r1) * invDet;
	return Matrix3x3(c0, c1, c2);
}

inline __host__ __device__ Matrix3x3 makeRotation(float3 eulerAngles) {
	Matrix3x3 Ryaw = Matrix3x3::fromColumns(make_float3(cos(eulerAngles.y), 0.0f, sin(eulerAngles.y)), make_float3(0.0f, 1.0f, 0.0f), make_float3(-sin(eulerAngles.y), 0.0f, cos(eulerAngles.y)));
	Matrix3x3 Rpitch = Matrix3x3::fromColumns(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, cos(eulerAngles.x), -sin(eulerAngles.x)), make_float3(0.0f, sin(eulerAngles.x), cos(eulerAngles.x)));
	Matrix3x3 Rroll = Matrix3x3::fromColumns(make_float3(cos(eulerAngles.z), -sin(eulerAngles.z), 0.0f), make_float3(sin(eulerAngles.z), cos(eulerAngles.z), 0.0f), make_float3(0.0f, 0.0f, 1.0f));

	return Rroll * Rpitch * Ryaw;
}


inline void printf(Matrix3x3 m)
{
	printf("%f %f %f\n", m.r0.x, m.r0.y, m.r0.z);
	printf("%f %f %f\n", m.r1.x, m.r1.y, m.r1.z);
	printf("%f %f %f\n", m.r2.x, m.r2.y, m.r2.z);
}

struct Matrix4x4 {
	float4 r0;
	float4 r1;
	float4 r2;
	float4 r3;

	__host__ __device__ Matrix4x4(float4 r0, float4 r1, float4 r2, float4 r3) : r0(r0), r1(r1), r2(r2), r3(r3) {}

	__host__ __device__ static Matrix4x4 fromColumns(float4 c0, float4 c1, float4 c2, float4 c3) {
		return Matrix4x4(make_float4(c0.x, c1.x, c2.x, c3.x), make_float4(c0.y, c1.y, c2.y, c3.y), make_float4(c0.z, c1.z, c2.z, c3.z), make_float4(c0.w, c1.w, c2.w, c3.w));
	}

	__host__ __device__ static Matrix4x4 Identity() {
		return Matrix4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 1.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 1.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 1.0f));
	}


};

inline void printf(Matrix4x4 m) {
	printf("%f %f %f %f\n", m.r0.x, m.r0.y, m.r0.z, m.r0.w);
	printf("%f %f %f %f\n", m.r1.x, m.r1.y, m.r1.z, m.r1.w);
	printf("%f %f %f %f\n", m.r2.x, m.r2.y, m.r2.z, m.r2.w);
	printf("%f %f %f %f\n", m.r3.x, m.r3.y, m.r3.z, m.r3.w);
}


inline __host__ __device__ Matrix4x4 operator*(const Matrix4x4& m, const float s) {
	return Matrix4x4(m.r0 * s, m.r1 * s, m.r2 * s, m.r3 * s);
}

inline __host__ __device__ Matrix4x4 operator*(const float s, const Matrix4x4& m) {
	return Matrix4x4(m.r0 * s, m.r1 * s, m.r2 * s, m.r3 * s);
}

inline __host__ __device__ Matrix4x4 operator*(const Matrix4x4& a, const Matrix4x4& b) {
	return Matrix4x4(
		make_float4(dot(make_float4(a.r0.x, a.r1.x, a.r2.x, a.r3.x), b.r0), dot(make_float4(a.r0.x, a.r1.x, a.r2.x, a.r3.x), b.r1), dot(make_float4(a.r0.x, a.r1.x, a.r2.x, a.r3.x), b.r2), dot(make_float4(a.r0.x, a.r1.x, a.r2.x, a.r3.x), b.r3)),
		make_float4(dot(make_float4(a.r0.y, a.r1.y, a.r2.y, a.r3.y), b.r0), dot(make_float4(a.r0.y, a.r1.y, a.r2.y, a.r3.y), b.r1), dot(make_float4(a.r0.y, a.r1.y, a.r2.y, a.r3.y), b.r2), dot(make_float4(a.r0.y, a.r1.y, a.r2.y, a.r3.y), b.r3)),
		make_float4(dot(make_float4(a.r0.z, a.r1.z, a.r2.z, a.r3.z), b.r0), dot(make_float4(a.r0.z, a.r1.z, a.r2.z, a.r3.z), b.r1), dot(make_float4(a.r0.z, a.r1.z, a.r2.z, a.r3.z), b.r2), dot(make_float4(a.r0.z, a.r1.z, a.r2.z, a.r3.z), b.r3)),
		make_float4(dot(make_float4(a.r0.w, a.r1.w, a.r2.w, a.r3.w), b.r0), dot(make_float4(a.r0.w, a.r1.w, a.r2.w, a.r3.w), b.r1), dot(make_float4(a.r0.w, a.r1.w, a.r2.w, a.r3.w), b.r2), dot(make_float4(a.r0.w, a.r1.w, a.r2.w, a.r3.w), b.r3))
	);
}
inline __host__ __device__ float4 operator*(const Matrix4x4& m, const float4& v) {
	return make_float4(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}

inline __host__ __device__ float3 operator*(const Matrix4x4& m, const float3& v) {
	float4 v4 = make_float4(v.x, v.y, v.z, 1.0f);
	float4 res = m * v4;
	return make_float3(res.x, res.y, res.z);
}

inline __host__ __device__ Matrix4x4 operator+(const Matrix4x4& a, const Matrix4x4& b) {
	return Matrix4x4(a.r0 + b.r0, a.r1 + b.r1, a.r2 + b.r2, a.r3 + b.r3);
}

inline __host__ __device__ Matrix4x4 operator-(const Matrix4x4& a, const Matrix4x4& b) {
	return Matrix4x4(a.r0 - b.r0, a.r1 - b.r1, a.r2 - b.r2, a.r3 - b.r3);
}

inline __host__ __device__ Matrix4x4 operator-(const Matrix4x4& a) {
	return Matrix4x4(-a.r0, -a.r1, -a.r2, -a.r3);
}

inline __host__ __device__ Matrix4x4 transpose(const Matrix4x4& m) {
	return Matrix4x4(make_float4(m.r0.x, m.r1.x, m.r2.x, m.r3.x), make_float4(m.r0.y, m.r1.y, m.r2.y, m.r3.y), make_float4(m.r0.z, m.r1.z, m.r2.z, m.r3.z), make_float4(m.r0.w, m.r1.w, m.r2.w, m.r3.w));
}

inline __host__ __device__ bool isTransform(const Matrix4x4& m) {
	return m.r0.w == 0.0f && m.r1.w == 0.0f && m.r2.w == 0.0f && m.r3.w == 1.0f;
}

inline __host__ __device__ Matrix3x3 extractR(const Matrix4x4& m) {
	return Matrix3x3(make_float3(m.r0.x, m.r0.y, m.r0.z), make_float3(m.r1.x, m.r1.y, m.r1.z), make_float3(m.r2.x, m.r2.y, m.r2.z));
}

inline __host__ __device__ float3 extractT(const Matrix4x4& m) {
	return make_float3(m.r3.x, m.r3.y, m.r3.z);
}

inline __host__ __device__  Matrix4x4 makeTransform(Matrix3x3 R, float3 t) {
	return Matrix4x4(make_float4(R.r0.x, R.r0.y, R.r0.z, t.x), make_float4(R.r1.x, R.r1.y, R.r1.z, t.y), make_float4(R.r2.x, R.r2.y, R.r2.z, t.z), make_float4(0.0f, 0.0f, 0.0f, 1.0f));
}

inline __host__ __device__ Matrix4x4 inverseTransform(const Matrix4x4& m) {
	Matrix3x3 R = extractR(m);
	float3 t = extractT(m);
	Matrix3x3 Rinv = transpose(R);
	float3 tinv = -Rinv * t;
	return makeTransform(Rinv, tinv);
}
// PATH TRACING UTILS


/*
Struct to represent a ray of light.
*/
struct Ray {
	float3 origin;
	float3 direction;

	__host__ __device__ Ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	__host__ __device__ float3 at(const float t) const
	{
		return origin + direction * t;
	}

};

struct HitRecord {
	float3 point;
	float3 normal;
	float t;
};

struct Camera {
	float3 position;
	float3 lookAt;
	float3 direction;
	float3 up;
	float3 right;
	float3 worldUp;
	float fov;
	int width;
	int height;

	__host__ __device__ Camera(float3 position, float3 lookAt, float3 worldUp, float fov, int width, int height) : position(position), lookAt(lookAt), worldUp(worldUp), fov(fov), width(width), height(height) {
		direction = normalize(lookAt - position);
		right = normalize(cross(direction, worldUp));
		up = normalize(cross(right, direction));
	}

	__host__ __device__ Ray getRay(float u, float v) {
		float3 center = position;
		float h = tan(fov * PI / 360.0f);
		float aspectRatio = (float)width / (float)height;
		float viewportHeight = 2.0f * h;
		float viewportWidth = aspectRatio * viewportHeight;
		float3 viewportU = right * viewportWidth;
		float3 viewportV = up * viewportHeight;
		float3 viewportOrigin = center - viewportU * 0.5f - viewportV * 0.5f - direction;
		return Ray(position, normalize(viewportOrigin + viewportU * (1-u) + viewportV * (1.0f-v) - position));

	}

	__host__ __device__ void setDirection(float3 newDirection) {
		direction = newDirection;
		right = normalize(cross(direction, worldUp));
		up = normalize(cross(right, direction));
	}

	__host__ __device__ void setPosition(float3 newPosition) {
		position = newPosition;
	}

	__host__ __device__ void setLookAt(float3 newLookAt) {
		lookAt = newLookAt;
		direction = normalize(lookAt - position);
		right = normalize(cross(direction, worldUp));
		up = normalize(cross(right, direction));
	}	

};

struct Sphere
{
	float3 center;
	float radius;

	__host__ __device__ Sphere(float3 center, float radius) : center(center), radius(radius) {}
};

bool __device__ intersectSphere(const Ray& ray, const Sphere& sphere, HitRecord& record);

struct SimpleBRDF {
	float roughness;
	float3 albedo;
	float metalness;

	__host__ __device__ SimpleBRDF(float3 albedo, float roughness, float metalness) : albedo(albedo), roughness(roughness), metalness(metalness) {}
};

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
	std::vector<int3> faces;
	std::vector<float3> vertexNormals;
	std::vector<float2> vertexUVs;
	float3 AABB[2];
};

bool __device__ intersectTriangle(const Ray& ray, const float3& edge1, const float3& edge2, HitRecord& record);

void loadObj(const char* filename, Mesh& mesh, float scalefactor=1.0f);


/*float& t
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

