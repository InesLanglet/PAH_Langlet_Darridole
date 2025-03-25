#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Gaussian function
__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}


// CUDA bilateral filter kernel
__global__ void bilateral_filter_cuda(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, float sigma_color, float sigma_space) {
    //Chaque thread CUDA s’occupe d’un pixel unique
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = d / 2;

    float filtered_value[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum[3] = {0.0f, 0.0f, 0.0f};

    unsigned char *center_pixel = src + (y * width + x) * channels;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                for (int c = 0; c < channels; c++) {
                    float spatial_weight = gaussian(sqrtf((float)(i * i + j * j)), sigma_space);
                    float range_weight = gaussian(fabsf((float)neighbor_pixel[c] - (float)center_pixel[c]), sigma_color);
                    float weight = spatial_weight * range_weight;

                    filtered_value[c] += neighbor_pixel[c] * weight;
                    weight_sum[c] += weight;
                }
            }
        }
    }

    unsigned char *output_pixel = dst + (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6f));
    }
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);

    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);

    cudaMemcpy(d_src, image, width * height * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(32,32);
    //dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    bilateral_filter_cuda<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, 5, 15.0f, 5.0f);
    cudaDeviceSynchronize();  // attend la fin du kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);

    stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels);

    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(image);
    free(filtered_image);

    printf("CUDA bilateral filter complete. Output saved as %s\n", argv[2]);
    return 0;
}