#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define DIMENSIONS 2
#define MAX_ITERATIONS 1000

typedef struct {
    double x;
    double y;
} Point;

double euclidean_distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void read_points_from_file(const char *filename, int num_points, Point *points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Unable to open file %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < num_points; i++) {
        fscanf(file, "%lf %lf", &points[i].x, &points[i].y);
    }
    fclose(file);
}

void k_means_clustering(const char *filename, int num_points, Point *points, int num_clusters, int rank, int size) {
    Point centroids[num_clusters];
    Point *all_centroids = NULL;
    int num_local_points = num_points / size;
    Point *local_points = (Point *)malloc(num_local_points * sizeof(Point));

    MPI_Scatter(points, num_local_points * DIMENSIONS, MPI_DOUBLE, local_points, num_local_points * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_clusters; i++) {
            centroids[i] = points[i];
        }
        all_centroids = (Point *)malloc(num_clusters * sizeof(Point) * size);
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        double *sum_x = calloc(num_clusters, sizeof(double));
        double *sum_y = calloc(num_clusters, sizeof(double));
        int *counts = calloc(num_clusters, sizeof(int));

        for (int i = 0; i < num_local_points; i++) {
            double min_dist = euclidean_distance(local_points[i], centroids[0]);
            int closest = 0;
            for (int j = 1; j < num_clusters; j++) {
                double dist = euclidean_distance(local_points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = j;
                }
            }
            sum_x[closest] += local_points[i].x;
            sum_y[closest] += local_points[i].y;
            counts[closest]++;
        }

        double *recv_sum_x = rank == 0 ? malloc(num_clusters * sizeof(double)) : NULL;
        double *recv_sum_y = rank == 0 ? malloc(num_clusters * sizeof(double)) : NULL;
        int *recv_counts = rank == 0 ? malloc(num_clusters * sizeof(int)) : NULL;

        MPI_Reduce(sum_x, recv_sum_x, num_clusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(sum_y, recv_sum_y, num_clusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(counts, recv_counts, num_clusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < num_clusters; i++) {
                centroids[i].x = recv_sum_x[i] / recv_counts[i];
                centroids[i].y = recv_sum_y[i] / recv_counts[i];
            }
            free(recv_sum_x);
            free(recv_sum_y);
            free(recv_counts);
        }
        MPI_Bcast(centroids, num_clusters * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(sum_x);
        free(sum_y);
        free(counts);
    }
    if (rank == 0) {
        printf("Final Centroids for %s:\n", filename);
        for (int i = 0; i < num_clusters; i++) {
            printf("Centroid %d: %.4lf %.4lf\n", i + 1, centroids[i].x, centroids[i].y);
        }
        free(all_centroids);
    }
    free(local_points);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    char filename[] = "points_50_000.txt";
    int num = 50000;
    Point *points = NULL;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        points = (Point *)malloc(num * sizeof(Point));
        if (!points) {
            fprintf(stderr, "Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_points_from_file(filename, num, points);
    }
    clock_t start_time = clock();
    k_means_clustering(filename, num, points, 10, rank, size); 
	clock_t end_time = clock();
    if (rank == 0) {
        free(points);
        double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    	printf("Execution Time for %s: %f seconds\n", filename, execution_time);
    }
    MPI_Finalize();
    return 0;
}
