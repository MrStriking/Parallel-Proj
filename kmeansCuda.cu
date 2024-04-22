%%writefile kmeansCuda.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define D 2         // Dimension of points
#define K 10        // Number of clusters
#define TPB 32      // Number of threads per block

// Euclidean distance of two 2D points
__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

__global__ void kMeansClusterAssignment(float* d_datapoints, int* d_clust_assn, float* d_centroids, int N) {
    //get idx for this datapoint
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= N) return;

    //find the closest centroid to this datapoint
    float min_dist = FLT_MAX;
    int closest_centroid = -1;

    for (int c = 0; c < K; ++c)
    {
        float dist = distance(d_datapoints[2 * idx], d_datapoints[2 * idx + 1], d_centroids[2 * c], d_centroids[2 * c + 1]);

        // Update of new cluster if it's closer
        if (dist < min_dist)
        {
            min_dist = dist;
            closest_centroid = c;
        }
    }
    //assign closest cluster id for this datapoint/thread
    d_clust_assn[idx] = closest_centroid;
}

// updating the new centroids according to the mean value of all the assigned data points
__global__ void kMeansCentroidUpdate(float* h_datapoints, int* h_clust_assn, float* h_centroids, int* h_clust_sizes, int N, int k) {

    float clust_datapoint_sums[2 * K] = { 0 };

    for (int j = 0; j < N; ++j) {
        // clust_id represents a cluster from 1...K
        int clust_id = h_clust_assn[j];
        clust_datapoint_sums[2 * clust_id] += h_datapoints[2 * j];
        clust_datapoint_sums[2 * clust_id + 1] += h_datapoints[2 * j + 1];
        h_clust_sizes[clust_id] += 1;
    }

    //Division by size (arithmetic mean) to compute the actual centroids
    for (int idx = 0; idx < K; idx++) {
        if (h_clust_sizes[idx])
        {
            h_centroids[2 * idx] = clust_datapoint_sums[2 * idx] / h_clust_sizes[idx];
            h_centroids[2 * idx + 1] = clust_datapoint_sums[2 * idx + 1] / h_clust_sizes[idx];
        }
    }

}

void read_points_from_file(const char *filename, float *h_datapoints, int num_points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Unable to open file %s.\n", filename);
        exit(1);
    }

    // Read points from file
    for (int i = 0; i < num_points; i++) {
        float x, y;
        if (fscanf(file, "%f %f", &x, &y) != 2) {
            fprintf(stderr, "Error reading from file.\n");
            exit(1);
        }
        h_datapoints[2 * i] = x;
        h_datapoints[2 * i + 1] = y;
    }
    fclose(file);
}

void centroid_init(float* h_datapoints, float* h_centroids, int N) {
	//initalize centroids
	for (int i = 0; i < K; i++) {
		int temp = (N / K);
		int idx_r = rand() % temp;
		h_centroids[2 * i] = h_datapoints[(i * temp + idx_r)];
		h_centroids[2 * i + 1] = h_datapoints[(i * temp + idx_r) + 1];
	}
}

int main() {
    const char* filename;
    int N=400000, MAX_ITER;

    // Set input file and maximum iterations
    filename = "points_250_000.txt";
    MAX_ITER = 4000;

    // Allocate memory on the device
    float* d_datapoints = 0;
    int* d_clust_assn = 0;
    float* d_centroids = 0;
    int* d_clust_sizes = 0;
    cudaMalloc(&d_datapoints, D * N * sizeof(float));
    cudaMalloc(&d_clust_assn, N * sizeof(int));
    cudaMalloc(&d_centroids, D * K * sizeof(float));
    cudaMalloc(&d_clust_sizes, K * sizeof(float));

    // Allocation of memory on the host
    float* h_centroids = (float*)malloc(D * K * sizeof(float));
    float* h_datapoints = (float*)malloc(D * N * sizeof(float));
    int* h_clust_sizes = (int*)malloc(K * sizeof(int));
    int* h_clust_assn = (int*)malloc(N * sizeof(int));

    srand(5);

    read_points_from_file(filename, h_datapoints, N);

    centroid_init(h_datapoints, h_centroids, N);

    // Initialize centroids counter for each cluster
    for (int c = 0; c < K; ++c) {
        h_clust_sizes[c] = 0;
    }

    // Initialize datapoints and centroids
    // Call your function to read from file and initialize datapoints here
    // Call your function to initialize centroids here

    clock_t start_time = clock();

    for (int cur_iter = 0; cur_iter < MAX_ITER; cur_iter++) {
        kMeansClusterAssignment<<<(N + TPB - 1) / TPB, TPB>>>(d_datapoints, d_clust_assn, d_centroids, N);

        // Call kernel to reset and update centroids directly on the device
        kMeansCentroidUpdate<<<(K + TPB - 1) / TPB, TPB>>>(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, N, K);
    
        // Only necessary to transfer centroids if they need to be output or analyzed on the host
        if (cur_iter == MAX_ITER - 1) {
            cudaMemcpy(h_centroids, d_centroids, D * K * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    clock_t end_time = clock();

    // Calculate execution time
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print final centroids
    printf("Final Centroids:\n");
    for (int l = 1; l <= K; l++) {
        printf("Centroid %d: %f, %f\n", l, h_centroids[2 * l], h_centroids[2 * l + 1]);
    }

    // Print execution time
    printf("Execution Time: %f seconds\n", execution_time);

    // Freeing memory on host
    free(h_centroids);
    free(h_datapoints);
    free(h_clust_sizes);
    free(h_clust_assn);

    // Freeing memory on device
    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);

    return 0;
}
