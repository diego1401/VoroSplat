#!/bin/bash

# List of iteration values to try
regularize_from_values=(
    10_000
    # 0
    # 5_000
    # 15_000
)

# List of depth weights to try
weights=(
    0.1
    # 1.0
)

folder_name="depth_normal_experiments_normal_fixed"

# Iterate over all combinations
for regularize_from in "${regularize_from_values[@]}"; do

    for weight in "${weights[@]}"; do

        # Only depth loss
        echo "Running experiment with:"
        echo "regularize_from: $regularize_from"
        echo "color_weight: 0.0"
        echo "depth_weight: $weight"
        echo "normal_weight: 0"
        
        python train_along_mesh.py -c configs/mipnerf360_indoor.yaml \
            --experiment_name "$folder_name/reg_from_${regularize_from}_depth_weight_${weight}" \
            --mesh_regularization_from "$regularize_from" \
            --mesh_color_loss_weight 0.0 \
            --mesh_depth_loss_weight "$weight" \
            --mesh_normal_loss_weight 0.0
        
        echo "----------------------------------------"

        # Depth and normal loss
        echo "Running experiment with:"
        echo "regularize_from: $regularize_from"
        echo "color_weight: 0.0"
        echo "depth_weight: $weight"
        echo "normal_weight: $weight"
        
        python train_along_mesh.py -c configs/mipnerf360_indoor.yaml \
            --experiment_name "$folder_name/reg_from_${regularize_from}_depth_weight_${weight}_normal_weight_${weight}" \
            --mesh_regularization_from "$regularize_from" \
            --mesh_color_loss_weight 0.0 \
            --mesh_depth_loss_weight "$weight" \
            --mesh_normal_loss_weight "$weight"
        
        echo "----------------------------------------"

        # # Color , depth and normal loss
        # echo "Running experiment with:"
        # echo "regularize_from: $regularize_from"
        # echo "color_weight: $weight"
        # echo "depth_weight: $weight"
        # echo "normal_weight: $weight"
        
        # python train_along_mesh.py -c configs/mipnerf360_indoor.yaml \
        #     --experiment_name "$folder_name/reg_from_${regularize_from}_depth_weight_${weight}_normal_weight_${weight}_color_weight_${weight}" \
        #     --mesh_regularization_from "$regularize_from" \
        #     --mesh_color_loss_weight $weight \
        #     --mesh_depth_loss_weight $weight \
        #     --mesh_normal_loss_weight $weight
        
        # echo "----------------------------------------"
    done
done

# # No mesh regularization
# echo "Running experiment with:"
# echo "regularize_from: 25_000"
# echo "color_weight: 0.0"
# echo "depth_weight: 0.0"
# echo "normal_weight: 0.0"

# python train_along_mesh.py -c configs/mipnerf360_indoor.yaml \
#     --experiment_name "$folder_name/vanilla" \
#     --mesh_regularization_from 25_000 \
#     --mesh_color_loss_weight 0.0 \
#     --mesh_depth_loss_weight 0.0 \
#     --mesh_normal_loss_weight 0.0

# echo "----------------------------------------"
