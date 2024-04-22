#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h> 

#define MAX_ITERATIONS 100

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
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_points; i++) {
        fscanf(file, "%lf %lf", &points[i].x, &points[i].y);
    }
    fclose(file);
}

void initialize_centroids(Point *centroids, Point *points, int num_clusters, int num_points) {
    for (int i = 0; i < num_clusters; i++) {
        int rand_index = rand() % num_points;
        centroids[i] = points[rand_index];
    }
}


void k_means_clustering(const char *filename, int num_points, Point *points, int num_clusters) {
    Point *centroids = (Point *)malloc(num_clusters * sizeof(Point));
    if (!centroids) {
        fprintf(stderr, "Failed to allocate centroids.\n");
        return;
    }
    initialize_centroids(centroids, points, num_clusters, num_points);

    clock_t start_time = clock();

    Point *sums = (Point *)calloc(num_clusters, sizeof(Point));
    int *counts = (int *)calloc(num_clusters, sizeof(int));

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        memset(sums, 0, num_clusters * sizeof(Point));
        memset(counts, 0, num_clusters * sizeof(int));

        #pragma omp parallel for
        for (int i = 0; i < num_points; i++) {
            double min_dist = INFINITY;
            int closest = 0;
            for (int j = 0; j < num_clusters; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = j;
                }
            }
            #pragma omp critical
            {
                sums[closest].x += points[i].x;
                sums[closest].y += points[i].y;
                counts[closest]++;
            }
        }
        int converged = 1;
        #pragma omp parallel for
        for (int i = 0; i < num_clusters; i++) {
            if (counts[i]) {
                Point new_centroid = {sums[i].x / counts[i], sums[i].y / counts[i]};
                if (euclidean_distance(centroids[i], new_centroid) > 0.0001) {
                    centroids[i] = new_centroid;
                    converged = 0;
                }
            }
        }

        if (converged) {
            break;
        }
    }

    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Final Centroids for %s:\n", filename);
    for (int i = 0; i < num_clusters; i++) {
        printf("Centroid %d: %.4lf %.4lf\n", i + 1, centroids[i].x, centroids[i].y);
    }
    printf("Execution Time for %s: %f seconds\n", filename, execution_time);

    free(centroids);
    free(sums);
    free(counts);
}

int main() {
    const char *file_names[] = {
        "points_1_000.txt",
        "points_10_000.txt",
        "points_50_000.txt",
        "points_100_000.txt",
        "points_250_000.txt"
    };
    const int num_points[] = {1000, 10000, 50000, 100000, 
        400000
    };

    for (int f = 0; f < sizeof(file_names) / sizeof(file_names[0]); f++) {
        Point *points = (Point *)malloc(num_points[f] * sizeof(Point));
        if (!points) {
            fprintf(stderr, "Memory allocation failed.\n");
            continue;
        }
        read_points_from_file(file_names[f], num_points[f], points);
        k_means_clustering(file_names[f], num_points[f], points, 10); // Assuming 10 clusters
        free(points);
    }
    return 0;
}
