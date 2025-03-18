import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from torch.nn.functional import relu
from torchvision.transforms.functional import to_pil_image

# all_id_to_split_id = {data_handler.idx_map[i]:i for i in range(len(data_handler.idx_map))}
# split_id_to_all_id = {v:k for k,v in all_id_to_split_id.items()}

def create_image_to_points(data_handler, all_id_to_split_id):
    # Create dict mapping image_id -> point3D_ids
    image_to_points = {}
    for point_id, values in data_handler.point3D_id_to_images.items():
        for v in values:
            
            image_id, pixel_id = v
            if image_id not in all_id_to_split_id.keys(): continue

            split_image_id = all_id_to_split_id[image_id]
            if split_image_id not in image_to_points:
                image_to_points[split_image_id] = []
            image_to_points[split_image_id].append([point_id, pixel_id])
    return image_to_points

def get_points_in_image(data_handler, image_to_points, image_idx):
    # Get all points in image 1
    image_points = image_to_points[image_idx]

    points_in_image = []
    # coords_of_points_in_image_1 = []
    for i in range(len(image_points)):
        pt_id = image_points[i][0]
        point_id_in_image = data_handler.point3D_id_to_point3D_idx[pt_id]
        point_in_im = data_handler.points3D[point_id_in_image]
        points_in_image.append(point_in_im)
        # coords_of_points_in_image_1.append(image_points[i][1])
    # Turn to numpy
    points_in_image = np.array(points_in_image)
    # coords_of_points_in_image_1 = np.array(coords_of_points_in_image_1)

    return points_in_image

def get_points_in_uv_space(data_handler, points_in_image, image_idx):
    # Transform points_in_image to view space
    points_in_view_space = (torch.from_numpy(points_in_image) -  data_handler.c2ws[image_idx, :3, 3:4].T) @ data_handler.c2ws[image_idx, :3, :3]
    
    points_in_uv_space = points_in_view_space @ data_handler.K.T
    depth_sfm = points_in_uv_space[...,2].numpy().copy()
    # disparity_sfm = 1.0 / (1e-6 + depth_sfm)

    points_in_uv_space[:,0] = points_in_uv_space[:,0] / points_in_uv_space[:,2]
    points_in_uv_space[:,1] = points_in_uv_space[:,1] / points_in_uv_space[:,2]
    points_in_uv_space[:,2] = 1.0

    return points_in_uv_space, depth_sfm

def get_uv_points_in_image(data_handler, image_to_points, i):
    points_in_image = get_points_in_image(data_handler, image_to_points, i)
    points_in_uv_space, depth_sfm = get_points_in_uv_space(data_handler, points_in_image, i)
    return points_in_uv_space, depth_sfm

def get_depth_map(data_handler, image_idx):
    # Get depth map of image
    image = data_handler.rgbs[image_idx]
    # Get depth
    model_pretrained = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",use_fast=True,device='cuda')
    d_model_raw = relu(model_pretrained(to_pil_image(image.permute(2,0,1)))['predicted_depth'].cuda().squeeze())

    return d_model_raw

def get_gt_depth_labels(points_in_uv_space,d_model_raw):
    y_coords = points_in_uv_space[:, 0].long().clamp(0, d_model_raw.shape[1] - 1)
    x_coords = points_in_uv_space[:, 1].long().clamp(0, d_model_raw.shape[0] - 1)

    disparity_gt = d_model_raw[x_coords, y_coords].cpu().numpy()

    return disparity_gt

def get_scaling_factor(x, y):
    # solves y = ax + b
    # Solving the linear system to find the scaling factor
    sum_x = x.sum()
    sum_y = y.sum()
    sum_dot = (x * y).sum()
    sum_x_sq = (x ** 2).sum()
    n = x.shape[0]

    a = np.array([sum_dot,sum_y])
    M = np.array([[sum_x_sq, sum_x], [sum_x, n]])

    alpha, beta = a @ np.linalg.inv(M)
    return alpha, beta