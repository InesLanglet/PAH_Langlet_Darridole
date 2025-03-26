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
    //Ici Chaque thread CUDA s’occupe d’un pixel unique
    //Le code traite des images 2D donc conserver les coordonnées x et y est mieux adapté aux images que si on avait utilisé un vecteur
    //De plus en gardant x et y, on manipule directement les lignes et les colonnes ce qui correspond à l'organisation de la mémoire de l'image.
    //L'utilisation d'un index vectoriel serait plus utile si on avait fait des traitements simples pixel par pixel sans voisinage
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // Sécurité pour les bords 

    int radius = d / 2;

    float filtered_value[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum[3] = {0.0f, 0.0f, 0.0f};

    unsigned char *center_pixel = src + (y * width + x) * channels;
    // Parcours de la fenêtre autour du pixel 
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int nx = x + j;
            int ny = y + i;
            // On vérifie que le voisin est bien dans l'image 
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
    // Calcul du pixel final après normalisation 
    unsigned char *output_pixel = dst + (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6f));
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_image>\n", argv[0]);
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

    int d_values[] = {3, 5, 9};
    float sigma_s_values[] = {10.0f, 30.0f, 75.0f};
    float sigma_r_values[] = {10.0f, 30.0f, 75.0f};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start);
                bilateral_filter_cuda<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d_values[k], sigma_r_values[j], sigma_s_values[i]);
                cudaEventRecord(stop);

                cudaDeviceSynchronize();

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
                }

                float milliseconds = 0;
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliseconds, start, stop);

                cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);

                char output_name[150];
                sprintf(output_name, "output_gpu_d%d_ss%.1f_sr%.1f.png", d_values[k], sigma_s_values[i], sigma_r_values[j]);
                stbi_write_png(output_name, width, height, channels, filtered_image, width * channels);

                printf("d = %d, sigma_s = %.1f, sigma_r = %.1f, GPU time = %f ms\n", d_values[k], sigma_s_values[i], sigma_r_values[j], milliseconds);
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(image);
    free(filtered_image);

    printf("All GPU bilateral filters complete.\n");
    return 0;
}

/* // Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
    // Chargement de l'image
    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }
    // Allocation de mémoire CPU pour l'image filtré de la taille de l'image
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    // Allocation de mémoire GPU 
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    // Copie de l'image sur le GPU
    cudaMemcpy(d_src, image, width * height * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(32,32); // Taille de grille adaptée à une image 512x512
    // Grille de 32x32 blocs = 1024 blocs
    // Bloc de 16x16 = 256 threads
    // Sachant que 512x512 = 262 144
    // 1024 blocs x 256 threads = 262 144
    // Ce calcul permet de vérifier que la grille CUDA couvre entièrement l'image
    // Appel du kernel CUDA avec les bons paramètres

    
    // Chronométrage CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bilateral_filter_cuda<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, 5, 15.0f, 5.0f);
    cudaEventRecord(stop);

    cudaDeviceSynchronize(); // Attendre la fin du kernel



    // Vérification des erreurs cuda 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time on GPU: %f ms\n", milliseconds);

    // Copie du résultat du GPU vers le CPU
    cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    // Copie du résultat du GPU vers le CPU
    stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels);
    // Libération de la mémoire
    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(image);
    free(filtered_image);

    printf("CUDA bilateral filter complete. Output saved as %s\n", argv[2]);
    return 0;
} */