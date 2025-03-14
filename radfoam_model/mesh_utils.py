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
        return images, \
               fragments.zbuf, \
               v_normalized
    

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
    height, width = image.shape[:2]
    square_side = min(height, width)
    reshape_gt = torchvision.transforms.CenterCrop((square_side, square_side))(image.permute(2, 0, 1))
    reshape_gt = torchvision.transforms.Resize((new_h, new_w))(reshape_gt).permute(1, 2, 0)
    return reshape_gt

def render_mesh(v,f,feat,data_handler,return_only_image=False,size=(64,64),idx=0):
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
    ground_truth_image, ray_batch_idx, rotation, translation, height, width, fx, fy, cx, cy = get_camera_parameters_from_data_handler(data_handler,idx,device)
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
    
    new_h,new_w = size
    raster_settings = RasterizationSettings(
        image_size=(new_h,new_w),
        faces_per_pixel=10
    )

    # Define rasterizer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Define shader (Phong shading)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)

    # Create the MeshRenderer
    renderer = MeshRendererWithDepthAndNormal(rasterizer=rasterizer, shader=shader)
    
    # Reshape ground truth image
    reshape_gt = reshape_image(ground_truth_image, new_h, new_w)
    reshape_ray_batch_idx = reshape_image(ray_batch_idx, new_h, new_w)
    
    return (reshape_gt, reshape_ray_batch_idx), renderer(mesh,return_only_image=return_only_image,eps=1e-8)

def mesh_render_plot(image,depth,normal,ground_truth_image,depth_radfoam_vis,normal_radfoam_vis,filename):
    # Convert prediction image from RGBA to RGB
    pred_rgb = image[..., :3].cpu().detach().squeeze(0).numpy()
    pred_rgb = pred_rgb.clip(0, 1)  # Ensure valid range

    # Plot depth
    pred_depth = depth[:,:,:,:1].mean(-1,keepdim=True).cpu().detach().squeeze(0).numpy()

    # Normal
    pred_normal = normal[:,:,:,:1, :].mean(-2,keepdim=False).cpu().detach().squeeze(0).numpy()

    # Normalize to range [0,1] for visualization
    pred_normal_vis = (pred_normal + 1) / 2  
    pred_normal_vis = np.clip(pred_normal_vis, 0, 1)  # Ensure values stay in range

    # Normalize normal for visualization
    radfoam_normal_vis = normal_radfoam_vis.cpu().detach().squeeze(0).numpy()
    radfoam_normal_vis = (radfoam_normal_vis + 1) / 2
    radfoam_normal_vis = np.clip(radfoam_normal_vis, 0, 1)

    # Normalize depth for visualization
    depth_min, depth_max = pred_depth.min(), pred_depth.max()
    if depth_max > depth_min:  # Avoid division by zero
        pred_depth_vis = (pred_depth - depth_min) / (depth_max - depth_min)
    else:
        pred_depth_vis = np.zeros_like(pred_depth)

    pred_depth_vis = np.clip(pred_depth_vis, 0, 1)

    # Normalize depth_radfoam_vis for visualization so that values are between 0 and 1
    depth_radfoam_vis = depth_radfoam_vis.cpu().detach().squeeze(0).numpy()
    depth_radfoam_vis = (depth_radfoam_vis - depth_radfoam_vis.min()) / (depth_radfoam_vis.max() - depth_radfoam_vis.min())
    depth_radfoam_vis = np.clip(depth_radfoam_vis, 0, 1)

    # Ensure ground truth image is also in valid range
    if ground_truth_image is None:
        ground_truth_image = np.zeros_like(pred_rgb)
    elif isinstance(ground_truth_image,torch.Tensor):
        gt_rgb = ground_truth_image.numpy()
    else:
        gt_rgb = np.array(ground_truth_image, dtype=np.float32) / 255.0  # Convert to [0,1] range

    if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
        gt_rgb = np.array(Image.fromarray((gt_rgb * 255).astype(np.uint8)).resize(pred_rgb.shape[1::-1])) / 255.0

    # Create a figure with 3 rows: RGB, Depth, and Normal comparisons
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # RGB comparison
    axes[0, 0].imshow(pred_rgb)
    axes[0, 0].set_title("Predicted RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth RGB")
    axes[0, 1].axis("off")

    # Depth comparison
    axes[1, 0].imshow(pred_depth_vis, cmap="viridis")
    axes[1, 0].set_title("Predicted Depth")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(depth_radfoam_vis, cmap="viridis")
    axes[1, 1].set_title("Ground Truth Depth")
    axes[1, 1].axis("off")

    # Normal comparison
    axes[2, 0].imshow(pred_normal_vis)
    axes[2, 0].set_title("Predicted Normal")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(radfoam_normal_vis)
    axes[2, 1].set_title("Ground Truth Normal")
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(filename)

def compute_normal_from_depth(depth, K):
    """
    Compute surface normals from a depth map using PyTorch operations.
    
    Args:
        depth (torch.Tensor): Depth map tensor of shape (H, W)
        K (torch.Tensor): Camera intrinsics matrix
        
    Returns:
        torch.Tensor: Normal map tensor of shape (H, W, 3)
    """
    # Compute gradients in x and y directions
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

    # normalize to unit vector
    normal_unit = normal_cross / torch.norm(normal_cross, dim=-1, keepdim=True)
    # set default normal to [0, 0, 1] for invalid values
    invalid_mask = ~torch.isfinite(normal_unit).all(dim=-1)
    normal_unit[invalid_mask] = torch.tensor([0., 0., 1.], device=depth.device)

    return normal_unit
