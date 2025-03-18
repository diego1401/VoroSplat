import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision


from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, HardPhongShader,HardGouraudShader,SoftGouraudShader,
    PointLights, DirectionalLights,
    PerspectiveCameras, TexturesVertex
)

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.ops import interpolate_face_attributes


def triangle_case1(tets, values, points, features, alpha_f, normal=True):
    '''one vertex is marked inside (resp. outside), three are outside (resp. inside).'''
    ins = tets[values<=0 if normal else values<0].repeat_interleave(3)
    out = tets[values>0 if normal else values>=0]
    
    # interpolate point position
    v_ins = values[values<=0 if normal else values<0].repeat_interleave(3)
    v_out = values[values>0 if normal else values>=0]
    alpha_value = (v_out/(v_out-v_ins))[:, None]
    new_points = alpha_value*points[ins] + (1-alpha_value)*points[out]
    
    # interpolate features
    new_features = alpha_f*features[ins] + (1-alpha_f)*features[out]
    
    # create triangles
    new_tri = torch.arange(len(new_points), device=points.device).reshape(len(new_points)//3,3)
    return new_points, new_tri, new_features

def triangle_case2(tets, values, points, features, alpha_f):
    '''two vertices are marked inside, two are marked outside'''
    ins = tets[values<=0]
    out = tets[values>0]
    
    # interpolate point position
    v_ins = values[values<=0]
    v_out = values[values>0]
    
    a1 = (v_out[::2, None]/(v_out[::2, None]-v_ins[::2, None]))
    p1 = a1*points[ins][::2] + (1-a1)*points[out][::2]
    
    a2 = (v_out[1::2, None]/(v_out[1::2, None]-v_ins[1::2, None]))
    p2 = a2*points[ins][1::2] + (1-a2)*points[out][1::2]
    
    a3 = (v_out[1::2, None]/(v_out[1::2, None]-v_ins[::2, None]))
    p3 = a3*points[ins][::2] + (1-a3)*points[out][1::2]
    
    a4 = (v_out[::2, None]/(v_out[::2, None]-v_ins[1::2, None]))
    p4 = a4*points[ins][1::2] + (1-a4)*points[out][::2]
    
    new_points = torch.cat((p1,p2,p3,p4))

    # interpolate features
    f1 = alpha_f*features[ins][::2] + (1-alpha_f)*features[out][::2]
    f2 = alpha_f*features[ins][1::2] + (1-alpha_f)*features[out][1::2]
    f3 = alpha_f*features[ins][::2] + (1-alpha_f)*features[out][1::2]
    f4 = alpha_f*features[ins][1::2] + (1-alpha_f)*features[out][::2]
    
    new_features = torch.cat((f1,f2,f3,f4))
    
    # create triangles
    ls = len(p1)
    new_tri = torch.tensor([[0,2*ls,3*ls], [1*ls,3*ls,2*ls]], device=points.device).repeat(ls,1)
    new_tri += torch.arange(ls, device=points.device).repeat_interleave(2)[:, None]
    return new_points, new_tri, new_features

def reverse_triangles(tri, reverse):
    tri[reverse, 0], tri[reverse, 1] = tri[reverse, 1].clone(), tri[reverse, 0].clone()

def marching_tetrahedra(tets, sdf_values, points, features, ret_edge = False):
    """
        marching tetrahedra of a given tet grid (in our case extracted from delaunay)

        Parameters:
            tets: (N, 4) tetrahedra indices
            sdf_values: (M, ) sdf tensor
            points: (M, 3) tensor of vertices
            features: (M, D) tensor of features at the vertices
            
        Returns:
            vertices: (P, 3)
            triangles: (Q, 3)
            interpolated features: (P, D)
            (optional) edges: (P, 2) indices of the original points/features
    """
    values = sdf_values[tets]
    
    pos = (values>0).sum(1)
    new_f = []
    
    # Compute triangle connectivity
    for i in [1, 2, 3]:
        if (pos==i).sum()>0:
            if i==1:
                reverse = torch.logical_or(values[:, 1]>0, values[:, 3]>0)[pos==1]
                out = tets[pos==1][values[pos==1]>0].repeat_interleave(3)
                ins = tets[pos==1][values[pos==1]<=0]
                new_tri = torch.column_stack((ins, out)).reshape(len(ins)//3,3,2)
                reverse_triangles(new_tri, reverse)
            if i==2:
                f13 = torch.logical_and(values[:, 1]<=0, values[:, 3]<=0)
                f02 = torch.logical_and(values[:, 0]<=0, values[:, 2]<=0)
                reverse = torch.logical_not(f13+f02)[pos==2]
                ins = tets[pos==2][values[pos==2]<=0]
                out = tets[pos==2][values[pos==2]>0]
                p1 = torch.column_stack((ins[::2], out[::2]))
                p2 = torch.column_stack((ins[1::2], out[1::2]))
                p3 = torch.column_stack((ins[::2], out[1::2]))
                p4 = torch.column_stack((ins[1::2], out[::2]))
                ps = torch.cat((p1, p2, p3, p4))
                ls = len(p1)
                new_tri = torch.tensor([[0,2*ls,3*ls], [1*ls,3*ls,2*ls]], device=points.device).repeat(ls,1)
                new_tri += torch.arange(ls, device=points.device).repeat_interleave(2)[:, None]
                new_tri = ps[new_tri]
                reverse_triangles(new_tri, reverse.repeat_interleave(2))
            if i==3:
                reverse = torch.logical_or(values[:, 0]<=0, values[:, 2]<=0)[pos==3]
                out = tets[pos==3][values[pos==3]>0]
                ins = tets[pos==3][values[pos==3]<=0].repeat_interleave(3)
                new_tri = torch.column_stack((ins, out)).reshape(len(ins)//3,3,2)
                reverse_triangles(new_tri, reverse)
        new_f.append(new_tri)
        
    nt = torch.cat(new_f)
    lv = len(sdf_values)
    hash = lv*nt[..., 1]+nt[..., 0]
    idx, tri_to_idx = hash.unique(return_inverse=True)
    edge = torch.column_stack((idx%lv, idx//lv))

    v_ins = sdf_values[edge[..., 0].flatten()]
    v_out = sdf_values[edge[..., 1].flatten()]
    alpha_value = (v_out/(v_out-v_ins))[:, None]
    
    new_v = points[edge[..., 0].flatten()]*alpha_value+points[edge[..., 1].flatten()]*(1-alpha_value)
    new_cf = features[edge[..., 0].flatten()]*alpha_value+features[edge[..., 1].flatten()]*(1-alpha_value)
    
    if ret_edge:
        return new_v, tri_to_idx, new_cf, edge
    
    return new_v, tri_to_idx, new_cf

def colmap_to_pytorch3d(rotation,translation,device):
    '''
    Changes the rotation and translation from the colmap to the pytorch 3D convention.
    '''
    rotation = torch.stack([-rotation[:, 0], -rotation[:, 1], rotation[:, 2]], 1) # from RDF to Left-Up-Forward for Rotation
    new_c2w = torch.cat([rotation, translation], 1)
    bottom = torch.Tensor([[0,0,0,1]]).to(device)
    w2c = torch.linalg.inv(torch.cat((new_c2w, bottom), 0))
    rotation, translation = w2c[:3, :3].permute(1, 0), w2c[:3, 3]

    return rotation, translation

def get_camera_parameters_from_data_handler(datahandler,idx,device):
    width, height = datahandler.img_wh
    c2w = datahandler.c2ws[idx].to(device)
    ground_truth_image = datahandler.train_rgbs.view(datahandler.c2ws.shape[0],height,width,3)[idx]
    ray_batch_idx = datahandler.train_rays.view(datahandler.c2ws.shape[0],height,width,6)[idx]
    K = datahandler.K.to(device)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3:]

    return ground_truth_image,ray_batch_idx,rotation, translation, height, width, fx, fy, cx, cy

def nan_grad_hook_full(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and torch.isnan(grad).any():
            nan_indices = torch.nonzero(torch.isnan(grad), as_tuple=True)
            print(f"renderer NaN detected in gradients of {module.__class__.__name__} at indices: {nan_indices}")
            raise ValueError("Nan ground in renderer")

def phong_normal_shading(meshes, fragments):
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )

    return pixel_normals

class MeshRendererWithDepthAndNormal(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world,return_only_image=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        if return_only_image:
            return images
        
        normals = phong_normal_shading(meshes_world,fragments)
        norm = torch.norm(normals, dim=-1, keepdim=True)
        v_normalized = normals / norm
        # v_normalized = v_normalized[:,:,:,:,[1,0,2]]
        invalid_mask = ~torch.isfinite(v_normalized).all(dim=-1)
        v_normalized[invalid_mask] = torch.tensor([0., 0., 1.], device=v_normalized.device)
        
        return images.squeeze(0), \
               fragments.zbuf.squeeze(0), \
               v_normalized.squeeze(0).squeeze(-2)
    
def reshape_image(image, new_h, new_w):
    """
    Reshapes a ground truth image to a new size while maintaining aspect ratio.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (H, W, C)
        new_h (int): New height
        new_w (int): New width
        
    Returns:
        torch.Tensor: Reshaped image tensor of shape (new_h, new_w, C)
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    reshape_gt = torchvision.transforms.Resize((new_h, new_w))(image.permute(2, 0, 1)).permute(1, 2, 0)
    return reshape_gt

def render_mesh(v,f,feat,data_handler,return_only_image=False,downsample=4,idx=0, use_depth_anything=False):
    '''
    Render the mesh given extracted vertices v and faces f. 
    This function outputs an image
    '''
    # Loading the mesh
    device = v.device
    verts = v.unsqueeze(0)
    faces = f.unsqueeze(0).to(device)
    features = feat.unsqueeze(0)

    textures = TexturesVertex(verts_features=features)

    mesh = Meshes(verts=verts, faces=faces, textures=textures)
    # Get the camera parameters
    # ground_truth_image, rotation, translation, height, width, fx, fy, cx, cy = get_camera_parameters(idx)
    ground_truth_image, ray_batch_idx, \
        rotation, translation, \
            height, width, fx, fy, cx, cy = get_camera_parameters_from_data_handler(data_handler,idx,device)
    rotation, translation = colmap_to_pytorch3d(rotation,translation,device)

    cameras = PerspectiveCameras(device=device, 
                                 R=rotation.unsqueeze(0).to(device), 
                                 T=translation.unsqueeze(0).to(device),
                                 in_ndc=False,
                                 image_size=((height,width),),
                                 focal_length=((fx,fy),),
                                 principal_point=((cx,cy),))
    
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]],ambient_color=((1.0,1.0,1.0),))

    # Rasterization settings
    height, width = ground_truth_image.shape[:2]
    new_h,new_w = height//downsample, width//downsample

    raster_settings = RasterizationSettings(
        image_size=(new_h,new_w),
        faces_per_pixel=1,
        blur_radius=0,
    )

    # Define rasterizer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Define shader (Phong shading)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

    # Create the MeshRenderer
    renderer = MeshRendererWithDepthAndNormal(rasterizer=rasterizer, shader=shader)
    
    # Reshape ground truth image
    reshape_gt = reshape_image(ground_truth_image, new_h, new_w)
    reshape_ray_batch_idx = reshape_image(ray_batch_idx, new_h, new_w)
    if use_depth_anything:
        depth_model = data_handler.depth_maps[idx]
        normal_model = data_handler.normal_maps[idx]
        depth_model = reshape_image(depth_model, new_h, new_w)
        normal_model = reshape_image(normal_model, new_h, new_w)
    else:
        depth_model = None
        normal_model = None
    return (reshape_gt, reshape_ray_batch_idx, depth_model, normal_model), renderer(mesh,return_only_image=return_only_image,eps=1e-8)

def mesh_render_plot(image, depth, normal, ground_truth_image, depth_radfoam, normal_radfoam_vis, filename):
    def convert_to_numpy(tensor, squeeze_dims=None):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach().numpy()
            if squeeze_dims:
                tensor = tensor.squeeze(*squeeze_dims)
        return tensor

    def normalize_for_visualization(data, min_val=None, max_val=None):
        if min_val is None or max_val is None:
            min_val, max_val = data.min(), data.max()
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return np.zeros_like(data)

    def clip_to_valid_range(data):
        return np.clip(data, 0, 1)

    # Convert tensors to numpy arrays
    pred_rgb = convert_to_numpy(image[..., :3])
    if depth.ndim == 4:
        pred_depth = convert_to_numpy(depth[:, :, :, :1].mean(-1, keepdim=True), squeeze_dims=(0,))
    else:
        pred_depth = convert_to_numpy(depth)
    
    if normal.ndim == 4:
        pred_normal = convert_to_numpy(normal[:, :, :, :1, :].mean(-2, keepdim=False), squeeze_dims=(0,))
    else:
        pred_normal = convert_to_numpy(normal)
    pred_normal_vis = (pred_normal + 1) / 2

    radfoam_normal_vis = convert_to_numpy(normal_radfoam_vis)
    radfoam_normal_vis = (radfoam_normal_vis + 1) / 2

    depth_radfoam_vis = convert_to_numpy(depth_radfoam)
    depth_radfoam_vis = normalize_for_visualization(depth_radfoam_vis)

    # Normalize depth for visualization
    pred_depth_vis = normalize_for_visualization(pred_depth)

    # Ensure ground truth image is in valid range
    if ground_truth_image is None:
        gt_rgb = np.zeros_like(pred_rgb)
    else:
        gt_rgb = convert_to_numpy(ground_truth_image)# / 255.0
        if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
            gt_rgb = np.array(Image.fromarray((gt_rgb * 255).astype(np.uint8)).resize(pred_rgb.shape[1::-1])) / 255.0

    # Clip values to valid range
    pred_rgb = clip_to_valid_range(pred_rgb)
    gt_rgb = clip_to_valid_range(gt_rgb)
    pred_normal_vis = clip_to_valid_range(pred_normal_vis)
    radfoam_normal_vis = clip_to_valid_range(radfoam_normal_vis)
    pred_depth_vis = clip_to_valid_range(pred_depth_vis)
    depth_radfoam_vis = clip_to_valid_range(depth_radfoam_vis)

    # Compute error maps
    rgb_error = np.abs(pred_rgb - gt_rgb).mean(axis=-1)
    depth_error = np.abs(pred_depth_vis.squeeze() - depth_radfoam_vis.squeeze())
    dot_product = np.sum(pred_normal_vis * radfoam_normal_vis, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    normal_error = 1.0 - dot_product

    # Create a figure with 3 rows: RGB, Depth, and Normal comparisons
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # RGB comparison and error
    axes[0, 0].imshow(pred_rgb)
    axes[0, 0].set_title("Predicted RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth RGB")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_error, cmap='hot')
    axes[0, 2].set_title("RGB Error")
    axes[0, 2].axis("off")

    # Depth comparison and error
    axes[1, 0].imshow(pred_depth_vis, cmap="viridis")
    axes[1, 0].set_title("Predicted Depth")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(depth_radfoam_vis, cmap="viridis")
    axes[1, 1].set_title("Ground Truth Depth")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(depth_error, cmap='hot')
    axes[1, 2].set_title("Depth Error")
    axes[1, 2].axis("off")

    # Normal comparison and error
    axes[2, 0].imshow(pred_normal_vis)
    axes[2, 0].set_title("Predicted Normal")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(radfoam_normal_vis)
    axes[2, 1].set_title("Ground Truth Normal")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(normal_error, cmap='hot')
    axes[2, 2].set_title("Normal Error")
    axes[2, 2].axis("off")

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def depths_to_points(data_handler, depthmap, idx):
    c2w = (data_handler.c2ws[idx].T)#.inverse()
    W, H = data_handler.img_wh
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(data_handler, depth, idx):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(data_handler, depth, idx).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def compute_normal_from_depth(depth, data_handler,idx,use_lu=False):
    """
    Compute surface normals from a depth map using PyTorch operations.
    
    Args:
        depth (torch.Tensor): Depth map tensor of shape (H, W)
        data_handler: Data handler object containing camera parameters
        idx (int): Index of the camera pose to use
        
    Returns:
        torch.Tensor: Normal map tensor of shape (H, W, 3) in world coordinates
    """
    # Compute gradients in x and y directions
    K = data_handler.K
    fx = K[0][0]
    fy = K[1][1]

    dz_dv, dz_du = torch.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = torch.stack([-dz_dx, -dz_dy, torch.ones_like(depth)], dim=-1)

    # Transform normal from image space to world space using the data_handler
    c2w = data_handler.c2ws[idx].cuda()[:3,:3].T

    if use_lu:
        raise NotImplementedError("LU decomposition is not implemented")
        lu = c2w.lu()
        normal_cross = normal_cross.view(-1,3).T.lu_solve(*lu)
        normal_cross = normal_cross.T.view(*depth.shape[:],3)
    else:
        normal_cross = normal_cross @ c2w

    # normalize to unit vector
    normal_unit = normal_cross / torch.norm(normal_cross, dim=-1, keepdim=True)

    # set default normal to [0, 0, 1] for invalid values
    invalid_mask = ~torch.isfinite(normal_unit).all(dim=-1)
    normal_unit[invalid_mask] = torch.tensor([0., 0., 1.], device=depth.device)

    return normal_unit
