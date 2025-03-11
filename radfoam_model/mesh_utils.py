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

def reverse_triangles(tri, reverse):
    tri[reverse, 0], tri[reverse, 1] = tri[reverse, 1].clone(), tri[reverse, 0].clone()

def marching_tetrahedra(tets, sdf_values, points, features, ret_edge = False):
    """
        marching tetrahedra of a given tet grid (in our case extracted from delaunay)

        Parameters:
            tets: (N,4) tetrahedra indices
            is_inside: (M,) boolean tensor, True if the point is inside the mesh
            points: (M,3) tensor of vertices
            features: (M,D) tensor of features at the vertices
            alpha_f: float between 0 and 1, interpolation factor for features. .5 is default, 1. is inside vertices only
            
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
    K = datahandler.K.to(device)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3:]

    return ground_truth_image,rotation, translation, height, width, fx, fy, cx, cy


def render_mesh(v,f,feat,data_handler,size=(64,64),idx=0):
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
    ground_truth_image, rotation, translation, height, width, fx, fy, cx, cy = get_camera_parameters_from_data_handler(data_handler,idx,device)
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
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # Define rasterizer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Define shader (Phong shading)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

    # Create the MeshRenderer
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render the image
    square_side = min(height,width)
    reshape_gt = torchvision.transforms.CenterCrop((square_side,square_side))(ground_truth_image.permute(2,0,1))
    reshape_gt = torchvision.transforms.Resize((new_h,new_w))(reshape_gt).permute(1,2,0)
    
    return reshape_gt,renderer(mesh)

def mesh_render_plot(image,ground_truth_image,filename):
    # Convert prediction image from RGBA to RGB
    pred_rgb = image[..., :3].cpu().detach().squeeze(0).numpy()
    pred_rgb = pred_rgb.clip(0, 1)  # Ensure valid range

    # Ensure ground truth image is also in valid range
    if ground_truth_image is None:
        ground_truth_image = np.zeros_like(pred_rgb)
    elif isinstance(ground_truth_image,torch.Tensor):
        gt_rgb = ground_truth_image.numpy()
    else:
        gt_rgb = np.array(ground_truth_image, dtype=np.float32) / 255.0  # Convert to [0,1] range

    if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
        gt_rgb = np.array(Image.fromarray((gt_rgb * 255).astype(np.uint8)).resize(pred_rgb.shape[1::-1])) / 255.0

    # Compute the error image (absolute difference)
    error_image = np.abs(pred_rgb - gt_rgb)
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot predicted image
    axes[0].imshow(pred_rgb)
    axes[0].set_title("Extraction")
    axes[0].axis("off")

    # Plot ground truth image
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Plot error image
    axes[2].imshow(error_image)
    axes[2].set_title("Error Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
