#pragma once
#include <vector>

#include "../utils/geometry.h"


namespace radfoam {

    /// @brief Apply Lloyd iterations to obtain a centroidal voronoi tessellation. TODO add n_lloyd_iterations as input
    void centroidal_voronoi_tessellation_single_iteration(ScalarType coord_scalar_type,
                                        const void *points,
                                        const void *point_adjacency,
                                        const void *point_adjacency_offsets,
                                        uint32_t num_points,
                                        void *updated_centroids,
                                        const void *stream = nullptr);
    
}