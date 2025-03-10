import torch

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
            
            

def render_mesh(v,f,feat,camera_parameters):

    # Device setup
    device = v.device

    # Example mesh (triangle)
    verts = v.unsqueeze(0)
    faces = f.unsqueeze(0).to(device)
    textures = TexturesVertex(verts_features=feat.unsqueeze(0))  # White color

    # Create a Meshes object
    mesh = Meshes(verts=verts, faces=faces, textures=textures)

    # Camera setup
    #TODO: import camera parameters

    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)

    cameras = PerspectiveCameras(device=device, R=R, T=T)
    # Lighting; TODO: Is this only ambient lighting ? 
    lights = PointLights(device=device, location=[[0, 0, 0]],ambient_color=((1.0,1.0,1.0),))

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=256,
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
    return renderer(mesh)