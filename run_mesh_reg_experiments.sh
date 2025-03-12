#!/bin/bash

# List of iteration values
regularize_from_iterations_list=(0 5000 10000 15000 25000)

# Loop through each iteration value
for regularize_from in "${regularize_from_iterations_list[@]}"; do
    echo "Running with mesh_regularization_from=$regularize_from"
    
    python train_along_mesh.py -c configs/mipnerf360_indoor.yaml \
        --experiment_name mesh_regularization_experiments/mesh_regularization_from_"$regularize_from" \
        --mesh_regularization_from "$regularize_from"
    
    echo "Finished iteration with mesh_regularization_from=$regularize_from"
    echo "-----------------------------------------"
done