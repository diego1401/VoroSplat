import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, HardPhongShader,HardGouraudShader,SoftGouraudShader,
    PointLights, DirectionalLights,
    PerspectiveCameras, TexturesVertex
)

from pytorch3d.renderer import look_at_view_transform
def triangle_case1(tets, values, points, features, alpha_f):
    '''one vertex is marked inside (resp. outside), three are outside (resp. inside).'''
    ins = tets[values<=0].repeat_interleave(3)
    out = tets[values>0]
    
    # interpolate point position
    v_ins = values[values<=0].repeat_interleave(3)
    v_out = values[values>0]
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

def marching_tetrahedra(tets, sdf_values, points, features, alpha_f=.5):
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
    new_v, new_f, new_cf = [], [], []
    cur_ind = 0
    for i in [1, 2, 3]:
        if (pos==i).sum()>0:
            if i==1:
                reverse = torch.logical_or(values[:, 1]>0, values[:, 3]>0)[pos==1]
                new_points, new_tri, new_features = triangle_case1(tets[pos==1], -values[pos==1], points, features, 1-alpha_f)
                reverse_triangles(new_tri, reverse)
            if i==2:
                f13 = torch.logical_and(values[:, 1]<0, values[:, 3]<0)
                f02 = torch.logical_and(values[:, 0]<0, values[:, 2]<0)
                reverse = torch.logical_not(f13+f02)[pos==2]
                new_points, new_tri, new_features = triangle_case2(tets[pos==2], values[pos==2], points, features, alpha_f)
                reverse_triangles(new_tri, reverse.repeat_interleave(2))
            if i==3:
                reverse = torch.logical_or(values[:, 0]<0, values[:, 2]<0)[pos==3]
                new_points, new_tri, new_features = triangle_case1(tets[pos==3], (values[pos==3]), points, features, alpha_f)
                reverse_triangles(new_tri, reverse)
            new_v.append(new_points)
            new_cf.append(new_features)
            new_f.append(cur_ind+new_tri)
            cur_ind += len(new_points)
    return torch.cat(new_v), torch.cat(new_f), torch.cat(new_cf)

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

def render_mesh(v,f,feat,datahandler,idx=0):
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
    ground_truth_image, rotation, translation, height, width, fx, fy, cx, cy \
        = get_camera_parameters_from_data_handler(datahandler,idx,device)
    
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
    raster_settings = RasterizationSettings(
        image_size=(height,width),
        blur_radius=0.0,
        faces_per_pixel=10
    )

    # Define rasterizer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Define shader (Phong shading)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

    # Create the MeshRenderer
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render the image
    return ground_truth_image,renderer(mesh).squeeze(0)

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
