// #pragma once
#include "centroidal_voronoi_tessellation.h"
#include "../utils/common_kernels.cuh"

// #include "delaunay.cuh"

namespace radfoam {

template <typename coord_scalar>
__global__ void
centroidal_voronoi_tessellation_single_iteration_kernel(const Vec3<coord_scalar> *__restrict__ points,
                         const uint32_t *point_adjacency,
                         const uint32_t *point_adjacency_offsets,
                         uint32_t num_points,
                         Vec3<coord_scalar> *__restrict__ updated_centroids) {
    uint32_t current_point_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (current_point_idx >= num_points) {
        return;
    }

    //TODO: Implement CVT
    // Get faces of associated primal point
    Vec3f primal_point = points[current_point_idx];
    uint32_t point_adjacency_begin = point_adjacency_offsets[current_point_idx];
    uint32_t point_adjacency_end = point_adjacency_offsets[current_point_idx + 1];

    uint32_t num_faces = point_adjacency_end - point_adjacency_begin;

    // Initialize updated centroids at 0
    updated_centroids[current_point_idx] = Vec3f::Zero();
    // Iterate over faces
    for (uint32_t face_number = 0; face_number < num_faces; face_number ++){
        uint32_t neighbor_idx = point_adjacency[point_adjacency_begin + face_number];
        Vec3f neighbor = points[neighbor_idx];
        
        updated_centroids[current_point_idx] += neighbor;
    }

    updated_centroids[current_point_idx] /= num_faces;

}

template <typename coord_scalar>
void centroidal_voronoi_tessellation_single_iteration(const Vec3<coord_scalar> *points,
                                     const uint32_t *point_adjacency,
                                     const uint32_t *point_adjacency_offsets,
                                     uint32_t num_points,
                                     Vec3<coord_scalar> *updated_centroids,
                                     const void *stream) {

    launch_kernel_1d<1024>(centroidal_voronoi_tessellation_single_iteration_kernel<coord_scalar>,
                        num_points,
                        stream,
                        points,
                        point_adjacency,
                        point_adjacency_offsets,
                        num_points,
                        updated_centroids);
}

void centroidal_voronoi_tessellation_single_iteration(ScalarType coord_scalar_type,
                                    const void *points,
                                    const void *point_adjacency,
                                    const void *point_adjacency_offsets,
                                    uint32_t num_points,
                                    void *updated_centroids,
                                    const void *stream){

    if (coord_scalar_type == ScalarType::Float32) {
        centroidal_voronoi_tessellation_single_iteration(
            static_cast<const Vec3<float> *>(points),
            static_cast<const uint32_t *>(point_adjacency),
            static_cast<const uint32_t *>(point_adjacency_offsets),
            num_points,
            static_cast<Vec3<float> *>(updated_centroids),
            stream);
    } else {
        throw std::runtime_error("unsupported scalar type");
    }

}

}
