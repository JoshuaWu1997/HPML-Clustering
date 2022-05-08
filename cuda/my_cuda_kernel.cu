#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TX 16
#define TY 16
#define TZ 32

#define TTX 32
#define TTY 32

__global__ void cdist(float *X, float *Y, float *Z, int p, int x_size, int y_size, int dim) {
    int x_0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y_0 = threadIdx.y + blockIdx.y * blockDim.y;
    int x_s = blockDim.x * gridDim.x;
    int y_s = blockDim.y * gridDim.y;

    for (int x = x_0; x < x_size; x += x_s)
        for (int y = y_0; y < y_size; y += y_s)
            for (int z = 0; z < dim; z += 1)
                Z[x * y_size + y] += powf(X[x * dim + z] - Y[y * dim + z], p);
}

__global__ void mdist(float *X, float *Y, float *Z, int p, int x_size, int y_size, int dim) {
    int x_0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y_0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z_0 = threadIdx.z + blockIdx.z * blockDim.z;
    int x_s = blockDim.x * gridDim.x;
    int y_s = blockDim.y * gridDim.y;
    int z_s = blockDim.z * gridDim.z;

    for (int x = x_0; x < x_size; x += x_s)
        for (int y = y_0; y < y_size; y += y_s)
            for (int z = z_0; z < dim; z += z_s)
                atomicAdd(Z + x * y_size + y, powf(fabsf(X[x * dim + z] - Y[y * dim + z]), p));
}

__host__ void mdist_interface(float *x, float *y, float *z, int p, int x_size, int y_size, int dim) {
    dim3 block3dSize((x_size + TX - 1) / TX, (y_size + TY - 1) / TY, (dim + TZ - 1) / TZ);
    dim3 grid3dSize(TX, TY, TZ);

    mdist<<<block3dSize, grid3dSize>>>(x, y, z, p, x_size, y_size, dim);
}

__host__ void cdist_interface(float *x, float *y, float *z, int p, int x_size, int y_size, int dim) {
    dim3 block2dSize((x_size + TTX - 1) / TTX, (y_size + TTY - 1) / TTY);
    dim3 grid2dSize(TTX, TTY);

    cdist<<<block2dSize, grid2dSize>>>(x, y, z, p, x_size, y_size, dim);
}