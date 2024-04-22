#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define DIMENSIONS 2
#define MAX_ITERATIONS 100

typedef struct {
    double x;
    double y;
} Point;

double euclidean_distance(Point a, Point b) {
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
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

void k_means_clustering(const char *filename, int num_points, Point *points, int num_clusters) {
    Point centroids[num_clusters];

    for (int i = 0; i < num_clusters; i++) {
        centroids[i].x = points[i].x;
        centroids[i].y = points[i].y;
    }

    clock_t start_time = clock();

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        int cluster_counts[num_clusters];
        Point sum[num_clusters];

        for (int i = 0; i < num_clusters; i++) {
            cluster_counts[i] = 0;
            sum[i].x = 0.0;
            sum[i].y = 0.0;
        }

        for (int i = 0; i < num_points; i++) {
            double min_dist = INFINITY;
            int closest_centroid = -1;

            for (int j = 0; j < num_clusters; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            if (closest_centroid != -1) {
                cluster_counts[closest_centroid]++;
                sum[closest_centroid].x += points[i].x;
                sum[closest_centroid].y += points[i].y;
            }
        }

        int converged = 1;
        for (int i = 0; i < num_clusters; i++) {
            if (cluster_counts[i] > 0) {
                Point new_centroid = {sum[i].x / cluster_counts[i], sum[i].y / cluster_counts[i]};
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

    printf("Final Centroids for %s:\n", filename);
    for (int i = 0; i < num_clusters; i++) {
        printf("Centroid %d: %.4lf %.4lf\n", i + 1, centroids[i].x, centroids[i].y);
    }

    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution Time for %s: %f seconds\n", filename, execution_time);
}

int main() {
    const char *file_names[] = {
        "points_1_000.txt",
        "points_10_000.txt",
        "points_50_000.txt",
        "points_100_000.txt",
        "points_250_000.txt"
    };

    const int num_points[] = {
        1000,
        10000,
        50000,
        100000,
        400000
    };

    for (int f = 0; f < sizeof(file_names) / sizeof(file_names[0]); f++) {
        char filename[50];
        strcat(filename, file_names[f]);

        int num = num_points[f];
        Point *points = (Point *)malloc(num * sizeof(Point));
        if (!points) {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }
        read_points_from_file(filename, num, points);
        k_means_clustering(filename, num, points, 10); 
        free(points);
    }
    return 0;
}
