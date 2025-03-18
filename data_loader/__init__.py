import os

import numpy as np
import einops
import torch
from torch.utils.data import DataLoader

import radfoam

from .colmap import COLMAPDataset
from .depth_map_extraction_utils import get_uv_points_in_image, create_image_to_points, get_depth_map, get_gt_depth_labels, get_scaling_factor
from tqdm import tqdm
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from radfoam_model.mesh_utils import compute_normal_from_depth
dataset_dict = {
    "colmap": COLMAPDataset,
}


def get_up(c2ws):
    right = c2ws[:, :3, 0]
    down = c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    A = torch.einsum("bi,bj->bij", right, right).sum(dim=0)
    A += torch.einsum("bi,bj->bij", forward, forward).sum(dim=0) * 0.02

    l, V = torch.linalg.eig(A)

    min_idx = torch.argmin(l.real)
    global_up = V[:, min_idx].real
    global_up *= torch.einsum("bi,i->b", -down, global_up).sum().sign()

    return global_up


def get_dataloader(
    dataset_args,
    split,
    rays_per_batch=65_536,
    downsample=None,
    shuffle_data=True,
):
    data_dir = os.path.join(dataset_args.data_path, dataset_args.scene)
    dataset = dataset_dict[dataset_args.dataset]
    if downsample is not None:
        split_dataset = dataset(
            data_dir, split=split, downsample=downsample, is_stack=True
        )
    else:
        split_dataset = dataset(data_dir, split=split, is_stack=True)

    if split == "train":
        n_rays = np.prod(split_dataset.all_rays.shape[:-1])
        inds = np.random.choice(n_rays, n_rays, replace=False)
        split_dataset.all_rays = split_dataset.all_rays.reshape(n_rays, 6)[inds]
        split_dataset.all_rgbs = split_dataset.all_rgbs.reshape(n_rays, 3)[inds]

        rem = n_rays % rays_per_batch
        split_dataset.all_rays = split_dataset.all_rays[: n_rays - rem]
        split_dataset.all_rgbs = split_dataset.all_rgbs[: n_rays - rem]

        split_dataset.all_rays = split_dataset.all_rays.reshape(
            -1, rays_per_batch, 6
        )
        split_dataset.all_rgbs = split_dataset.all_rgbs.reshape(
            -1, rays_per_batch, 3
        )

    dataloader = DataLoader(
        split_dataset,
        batch_size=1,
        shuffle=shuffle_data,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )

    if hasattr(split_dataset, "cameras") and split_dataset.cameras is not None:
        cameras = split_dataset.cameras
    else:
        cameras = None

    return dataloader, cameras


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)
        self.img_wh = None
        self.patch_size = 8

    def reload(self, split, downsample=None, use_depth_anything=False):
        data_dir = os.path.join(self.args.data_path, self.args.scene)
        dataset = dataset_dict[self.args.dataset]
        
        if downsample is not None:
            split_dataset = dataset(
                data_dir, split=split, downsample=downsample, is_stack=True
            )
        else:
            split_dataset = dataset(data_dir, split=split, is_stack=True)
        self.img_wh = split_dataset.img_wh
        self.c2ws = split_dataset.poses
        self.idx_map = split_dataset.idx_map
        self.K = split_dataset.intrinsics
        self.rays, self.rgbs = split_dataset.all_rays, split_dataset.all_rgbs
        self.cameras = split_dataset.cameras

        self.viewer_up = get_up(self.c2ws)
        self.viewer_pos = self.c2ws[0, :3, 3]
        self.viewer_forward = self.c2ws[0, :3, 2]

        try:
            self.points3D = split_dataset.points3D
            self.points3D_colors = split_dataset.points3D_color
            self.point3D_id_to_images = split_dataset.point3D_id_to_images
            # self.point3D_ids = split_dataset.point3D_ids
            self.point3D_id_to_point3D_idx = split_dataset.point3D_id_to_point3D_idx
        except:
            self.points3D = None
            self.points3D_colors = None
            self.point3D_id_to_images = None
            # self.point3D_ids = None
            self.point3D_id_to_point3D_idx = None

        if split == "train":

            if self.args.patch_based:
                dw = self.img_wh[0] - (self.img_wh[0] % self.patch_size)
                dh = self.img_wh[1] - (self.img_wh[1] % self.patch_size)
                w_inds = np.linspace(0, self.img_wh[0] - 1, dw, dtype=int)
                h_inds = np.linspace(0, self.img_wh[1] - 1, dh, dtype=int)

                self.train_rays = self.rays[:, h_inds, :, :]
                self.train_rays = self.train_rays[:, :, w_inds, :]
                self.train_rgbs = self.rgbs[:, h_inds, :, :]
                self.train_rgbs = self.train_rgbs[:, :, w_inds, :]

                self.train_rays = einops.rearrange(
                    self.train_rays,
                    "n (x ph) (y pw) r -> (n x y) ph pw r",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                self.train_rgbs = einops.rearrange(
                    self.train_rgbs,
                    "n (x ph) (y pw) c -> (n x y) ph pw c",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )

                self.batch_size = self.rays_per_batch // (self.patch_size**2)
            else:
                self.train_rays = einops.rearrange(
                    self.rays, "n h w r -> (n h w) r"
                )
                self.train_rgbs = einops.rearrange(
                    self.rgbs, "n h w c -> (n h w) c"
                )

                self.batch_size = self.rays_per_batch
            
            if use_depth_anything:
                self.depth_maps, self.normal_maps = self.get_depth_map_and_normal_map_from_pretrained()

    def get_depth_map_and_normal_map_from_pretrained(self):
        if os.getcwd().split('/')[-1] == 'experiments':
            data_dir = os.path.join('../depth_anything_maps', self.args.scene)
        else:
            data_dir = os.path.join('depth_anything_maps', self.args.scene)
        depth_maps_path = os.path.join(data_dir, "depth_maps_depthanything.npy")
        normal_maps_path = os.path.join(data_dir, "normal_maps_depthanything.npy")

        if os.path.exists(depth_maps_path) and os.path.exists(normal_maps_path):
            depth_maps = np.load(depth_maps_path, allow_pickle=True)
            normal_maps = np.load(normal_maps_path, allow_pickle=True)
        else:
            # Create depth_anything_maps folder if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            print("Computing depth maps and normal maps from pretrained model")
            depth_maps = []
            normal_maps = []
            all_id_to_split_id = {self.idx_map[i]: i for i in range(len(self.idx_map))}
            image_to_points = create_image_to_points(self, all_id_to_split_id)
            for i in tqdm(range(len(self.c2ws))):
                points_uv, D_sfm = get_uv_points_in_image(self, image_to_points, i)
                d_sfm = 1 / D_sfm
                d_raw = get_depth_map(self, i)
                d_gt = get_gt_depth_labels(points_uv, d_raw)
                alpha, beta = get_scaling_factor(d_gt, d_sfm)

                # Transform
                d = alpha * d_raw + beta
                D_gt = 1.0 / (1e-6 + d)

                normal = compute_normal_from_depth(D_gt, self, i)

                depth_maps.append(D_gt.cpu().numpy())
                normal_maps.append(normal.cpu().numpy())

            np.save(depth_maps_path, depth_maps)
            np.save(normal_maps_path, normal_maps)

        return depth_maps, normal_maps
            

    def get_iter(self):
        ray_batch_fetcher = radfoam.BatchFetcher(
            self.train_rays, self.batch_size, shuffle=True
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            self.train_rgbs, self.batch_size, shuffle=True
        )

        while True:
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()

            yield ray_batch, rgb_batch
