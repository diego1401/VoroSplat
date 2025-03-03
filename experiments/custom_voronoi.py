"""Custom Voronoi diagram based on PyTorch"""
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import igl
from scipy.spatial import Voronoi, Delaunay
from pytorch3d.ops import knn_points, knn_gather
from sklearn.neighbors import NearestNeighbors
import sys
import networkx as nx

try:
    sys.path.append("./cpp_utils/build")
    from VoroMeshUtils import compute_voromesh, self_intersect
    CPP_COMPILED = True
except:
    print('WARNING: CGAL voromesh not found, using scipy mesh extraction with NO WATERTIGHTNESS GUARANTEES. Please compile cpp_utils.')
    CPP_COMPILED = False

SIGNS = np.array(
    [
        [(-1) ** i, (-1) ** j, (-1) ** k]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)

class KNNResult:
    def __init__(self, dists, idx):
        self.dists = dists
        self.idx = idx

def knn( pc1: torch.Tensor,pc2: torch.Tensor, K: int,use_sklearn:bool=False):

    if use_sklearn:
        if pc1.dim() > 2:
            pc1 = pc1.view(-1, pc1.size(-1))
        if pc2.dim() > 2:
            pc2 = pc2.view(-1, pc2.size(-1))

        pc1_np = pc1.detach().cpu().numpy()
        pc2_np = pc2.detach().cpu().numpy()

        nn = NearestNeighbors(n_neighbors=K)
        nn.fit(pc2_np)

        distances_squared, indices = nn.kneighbors(pc1_np, return_distance=True)

        dists = torch.tensor(distances_squared, dtype=torch.float32,device=pc1.device)
        idx = torch.tensor(indices, dtype=torch.long,device=pc1.device)

        knn_pts=KNNResult(dists[None, :],idx[None, :])
    else:
        knn_pts=knn_points(pc1, pc2, K=K)

    return knn_pts

# utilities
def export_obj(nv: np.ndarray, nf: np.ndarray, name: str, nvn=None):
    if name[-4:] != ".obj":
        name += ".obj"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    # file.write("o {} \n".format(name))

    for v in nv:
        file.write("v {} {} {}\n".format(*v))
    file.write("\n")

    if nvn is not None:
        for vn in nvn:
            file.write("vn {} {} {}\n".format(*vn))
    file.write("\n")

    for face in nf:
        file.write("f " + " ".join([str(fi + 1) for fi in face]) + "\n")
    file.write("\n")

def export_ply_sh(nv: np.ndarray, nf: np.ndarray, nc: np.ndarray, name: str):
    """exports mesh (nv,nf) with SH coefficients (nc)"""
    if name[-4:] != ".ply":
        name += ".ply"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("element vertex {}\n".format(nv.shape[0]))
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    for i in range(nc.shape[1]):
        for j in range(nc.shape[2]):
             file.write("property float sh_{}_{}\n".format(i,j))
    file.write("element face {}\n".format(nf.shape[0]))
    file.write("property list uchar int vertex_indices\n")
    file.write("end_header\n")
    for i in range(len(nv)):
        file.write("{} {} {} ".format(*nv[i]) + ' '.join(nc[i].flatten().astype(str)))
        file.write("\n")
    for f in nf:
        file.write("3 {} {} {}\n".format(*f))
    file.close()

def face_orientation(p1, p2, p3, vp1, vp2):
    return (np.cross(p2 - p1, p3 - p1) * (vp1 - vp2)).sum() > 0


def mask_relevant_voxels(grid_n: int, samples: np.ndarray):
    """subselects voxels which collide with pointcloud"""

    samples_low = np.floor((samples + 1) / 2 * (grid_n - 1)).astype(np.int64)
    mask = np.zeros((grid_n, grid_n, grid_n))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mask[
                    samples_low[:, 0] + i, samples_low[:, 1] +
                    j, samples_low[:, 2] + k
                ] += 1
    return mask.reshape((grid_n**3)) > 0


def voronoi_to_poly_mesh(vpoints: np.ndarray, interior_cells: np.ndarray, clip=True, lims=np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])):
    """mesh voronoi diagram with marked interior cells"""
    
    pv = Voronoi(vpoints)
    faces = [0]

    prob_cnt = 0
    inds = []
    for face_, int1, int2 in zip(
        pv.ridge_vertices, pv.ridge_points[:, 0], pv.ridge_points[:, 1]
    ):
        face = face_.copy()
        if np.logical_xor(interior_cells[int1], interior_cells[int2]):
            if -1 in face:
                prob_cnt += 1
                print("WARNING: face ignored in the voronoi diagram")
            else:
                vp1 = pv.points[int1]
                vp2 = pv.points[int2]
                if interior_cells[int1]:
                    vp1, vp2 = vp2, vp1
                orient = face_orientation(
                    pv.vertices[face[0]],
                    pv.vertices[face[1]],
                    pv.vertices[face[2]],
                    vp1,
                    vp2,
                )
                faces.append(len(face) + faces[-1])

                if not (orient):
                    face.reverse()
                inds += face

    # select only the relevant vertices
    inds = np.array(inds)
    nvertices = pv.vertices.copy()
    un = np.unique(inds)
    inv = np.arange(inds.max() + 1)
    inv[un] = np.arange(len(un))
    nvertices = pv.vertices[un]
    inds = inv[inds]

    if clip:
        nvertices = np.maximum(lims[0].reshape(1, -1), nvertices)
        nvertices = np.minimum(lims[1].reshape(1, -1), nvertices)

    is_erronious = bool(prob_cnt > 0)

    return (
        nvertices,
        [inds[faces[i]: faces[i + 1]].tolist() for i in range(len(faces) - 1)],
        is_erronious,
    )


def voronoi_to_mesh(
    vpoints: np.ndarray, interior_cells: np.ndarray, clip=True, lims=np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]), return_errors=False
):
    """computes mesh from voronoi centers marked as inside or outside, with optional clip in [-1, 1]^3"""
    if CPP_COMPILED:
        vpoints = vpoints.astype(np.double)
        vpoints += np.random.randn(*vpoints.shape) * 1e-7
        interior_cells = (interior_cells-.5).astype(np.double)
        nvertices, nfaces = compute_voromesh(vpoints, interior_cells)
        return nvertices, nfaces
    else:
        nvertices, faces, is_erronious = voronoi_to_poly_mesh(
            vpoints, interior_cells, clip, lims)

        nfaces = []
        for face in faces:
            for i in range(2, len(face)):
                nfaces.append([face[0], face[i - 1], face[i]])

        if return_errors:
            return nvertices, np.array(nfaces), is_erronious
        else:
            return nvertices, np.array(nfaces)


def triangle_circumcenter(A, B, C):
    a = np.sqrt(((B-C)**2).sum(-1))
    b = np.sqrt(((A-C)**2).sum(-1))
    c = np.sqrt(((A-B)**2).sum(-1))
    acosA = a * a * ((C - A)*(B - A)).sum(-1)
    bcosB = b * b * ((A - B)*(C - B)).sum(-1)
    ccosC = c * c * ((B - C)*(A - C)).sum(-1)
    return (A * acosA[..., None] + B * bcosB[..., None] + C * ccosC[..., None]) / (acosA + bcosB + ccosC)[..., None]


def repulsion_force(points, K = 10, repulsion_fac = 10):
    # clamping
    points_to_points = knn(points[None, :], points[None, :], K=K+1)
    dists_to_neigh = points_to_points.dists.squeeze(0)[:, 1:]
    return torch.exp(- repulsion_fac * dists_to_neigh) / repulsion_fac

def attraction_force(points, K = 10, repulsion_fac = 10):
    points_to_points = knn(points[None, :], points[None, :], K=K+1)
    dists_to_neigh = points_to_points.dists.squeeze(0)[:, 1:]
    return dists_to_neigh

# model
class VoronoiValues(nn.Module):
    def __init__(self, points: np.ndarray = None, values: np.ndarray = None, points_opt: bool = True, values_opt: bool = False):
        super().__init__()
        if points is None:
            points = np.zeros((0, 3), dtype=np.float32)
        if values is None:
            values = np.zeros_like(points[:, 0])
        self.delaunay = None
        self.points_opt = points_opt
        self.values_opt = values_opt
        self.points = nn.Parameter(torch.tensor(points, dtype=torch.float32), requires_grad=points_opt)
        self.values_w = nn.Parameter(torch.tensor(values, dtype=torch.float32), requires_grad=values_opt)
        self.kedge = 10  # search for nearest face
        self.knn = 10  # for normalization
        self.repulsion_fac = 100  # for repulsion

    @property
    def values(self):
        return F.softplus(self.values_w)

    # Cells addition/removal
    def replace_cells(self, points, values_w, points_opt=None, values_opt=None):
        """resets all cells"""
        self.points = nn.Parameter(
            points, requires_grad=self.points_opt if (points_opt is None) else points_opt
        ).to(self.points.device)
        self.values_w = nn.Parameter(
            values_w, requires_grad=self.values_opt if (values_opt is None) else values_opt
        ).to(self.values.device)

    def subselect_cells(self, inds: np.ndarray, points_opt=None, values_opt=None):
        self.replace_cells(self.points[inds].clone().detach(), self.values_w[inds].clone().detach(),
                           points_opt=points_opt, values_opt=values_opt)

    def add_cells(self, points: np.ndarray, values: np.ndarray = None):
        if values is None:
            values = np.zeros_like(points[:, 0])
        device = self.points.device
        npoints = torch.cat(
            (self.points.detach(), torch.tensor(
                points, dtype=torch.float32).to(device))
        )
        nvalues = torch.cat(
            (self.values.detach(), torch.tensor(
                values, dtype=torch.float32).to(device))
        )
        self.replace_cells(npoints, nvalues)

    def closest_cells(self, points: torch.tensor, number=2):
        return knn(points[None, :], self.points[None, :], K=number).idx[0]

    def squared_distance_to_edges(self, points: torch.tensor, return_indices=False):
        """computes distance to closest voronoi face WARNING: relying on knn"""
        indices = self.closest_cells(points, self.kedge + 1)
        point_to_voronoi_center = points - self.points[indices[:, 0]]
        voronoi_edge = self.points[indices[:, 1:]] - \
            self.points[indices[:, 0, None]]
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, None, :] * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        if return_indices:
            sind = indices[torch.arange(len(sq_dist)), 1 + sq_dist.min(1)[1]]
            return torch.hstack((indices[:, 0, None], sind[:, None]))
        return sq_dist.min(1)[0]

    # Move cells
    def clamp(self):
        """clamps centers in bounding box"""
        with torch.no_grad():
            self.points[:] = torch.clamp(self.points, -1, 1)

    # Flag cells
    def select_relevant_cells(self, points: torch.tensor):
        """removes cells unactivated by the sample points"""
        with torch.no_grad():
            indices = self.squared_distance_to_edges(points, True)
            self.subselect_cells(torch.unique(indices))

    # Mesh
    def set_values_winding(self, v: np.ndarray, f: np.ndarray, barycenter=False):
        """sets values to winding number minus 0.5, from voronoi centers (default) or cell barycenters"""
        with torch.no_grad():
            if not (barycenter):
                self.values[:] = torch.tensor(
                    igl.fast_winding_number_for_meshes(
                        v, f, self.points.cpu().detach().numpy().astype("double")
                    )
                    - 0.5
                )
            else:
                vpoints = np.concatenate(
                    (self.points.cpu().detach().numpy(), SIGNS))
                pv = Voronoi(vpoints)
                vverts = []
                for e in pv.point_region:
                    delt = pv.vertices[pv.regions[e]]
                    vverts.append(delt.mean(0))
                self.vverts = np.row_stack(vverts)
                fws = igl.fast_winding_number_for_meshes(v, f, self.vverts)
                self.values[:] = torch.tensor(1.0 * (fws[:-8] > 0.5) - 0.5)

    def to_mesh(
        self, v: np.ndarray = None, f: np.ndarray = None,
        clip=True, lims=np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]),
        threshold=0., return_errors=False
    ):
        """extracts mesh
            WARNING: the output mesh can contain self-intersections due to numerical errors. For a self-intersection free mesh, use the CGAL version in cpp_utils"""
        if not v is None:
            self.set_values_winding(v, f)
        return voronoi_to_mesh(
            self.points.detach().cpu().numpy(),
            (self.values_w.detach() >= threshold).cpu().numpy().astype(int),
            clip=clip, lims=lims,
            return_errors=return_errors,
        )

    def clean_useless_generators(self, return_mask=False, points_opt=None, values_opt=None, threshold=0.,
                                 partial=False, partial_method='random', partial_ratio=0.5, knn=1):
        pv = Voronoi(self.points.detach().cpu().numpy())
        interior = (self.values_w.detach() >= threshold).cpu().numpy()
        keep = np.zeros(pv.points.shape[0], dtype=bool)
        keep_mask = pv.ridge_points[np.logical_xor(interior[pv.ridge_points[:, 0]], interior[pv.ridge_points[:, 1]])].flatten()

        keep[keep_mask] = True
        if partial:
            if partial_method == 'random':
                remove_idx = np.logical_not(keep_mask).nonzero()
                random_keep_idx = np.random.choice(remove_idx, size=int(partial_ratio * remove_idx.shape[0]), replace=False)
                keep[random_keep_idx] = True
            if partial_method == 'knn':
                with torch.no_grad():
                    k2r_knn = knn_points(self.points[keep].unsqueeze(0), self.points[np.logical_not(keep)].unsqueeze(0), K=knn)
                knn_keep_idx = np.arange(pv.points.shape[0])[np.logical_not(keep)][k2r_knn.idx.flatten().cpu().numpy()]
                keep[knn_keep_idx] = True

        self.subselect_cells(keep, points_opt=points_opt, values_opt=values_opt)
        return keep if return_mask else None
    


    def clean_small_inside_components(self):
        """remove floaters inside the shape (outside is <= 0)"""
        # build Voronoi diagram
        pv = Voronoi(self.points.cpu().detach().numpy())

        interior = (self.values <= 0).cpu().detach().numpy()

        # build connectivity graph
        network = nx.Graph()
        network.add_nodes_from(np.arange(len(self.points)))
        candidate_edges = pv.ridge_points
        candidate_edges = candidate_edges [interior[candidate_edges[:, 0]]*interior[candidate_edges[:, 1]]]
        network.add_edges_from(candidate_edges)

        # select seed: infinite Voronoi cell marked outside
        outside_seed = None
        for i, pr in enumerate(pv.point_region):
            if -1 in pv.regions[pr] and self.values[i]<=0:
                outside_seed = i
                break

        # the outside is now the connected component of this seed
        outside_mask = np.array(list(nx.node_connected_component(network, outside_seed)))

        # assign new values
        with torch.no_grad():
            self.values[:] = 1
            self.values[outside_mask] = 0

        
    def ARAP(self, moved_indices, moved_new_positions, fixed_indices=None, deformation_width=10):
        """deforms the tet grid of generators with ARAP constraints. Can be made faster by storing the Delaunay"""
        assert len(moved_indices)==len(moved_new_positions)
        points = self.points.cpu().detach().numpy().copy()
        
        # compute delaunay (slow)
        if self.delaunay is None:
            self.delaunay = Delaunay(points)
        else:
            print('using stored delaunay')
        # filter tets which are composed of "outside" generators
        filter_simp = (self.values_w[self.delaunay.simplices]>0).sum(-1)
        filter_simp = (filter_simp>0).detach().cpu().numpy()

        if fixed_indices is None:
            # breadth-first search with deformation_width iterations
            to_see = list(moved_indices)
            fixed = np.ones(len(points), dtype=bool)
            simplices_filtered = self.delaunay.simplices[filter_simp]
            for _ in range(deformation_width):
                new_to_see = []
                for point_idx in to_see:
                    if fixed[point_idx]:
                        fixed[point_idx] = False
                        new_to_see.append(np.unique(simplices_filtered[(simplices_filtered==point_idx).any(-1)]))
                to_see = np.concatenate(new_to_see)
            fixed_indices = np.arange(len(self.points))[fixed]

        # compute the ARAP deformation 
        arap = igl.ARAP(points, self.delaunay.simplices[filter_simp], 3, np.concatenate((fixed_indices, moved_indices)))
        constraints = np.concatenate((points[fixed_indices], moved_new_positions))
        new_points = arap.solve(constraints, points)
        return new_points



# Training
def train_voronoi(
    V: VoronoiValues,
    points: np.ndarray,
    optimizer: torch.optim.Optimizer,
    fac=1,
    clamp=True,
):
    optimizer.zero_grad()
    if fac < 1:
        mask = torch.rand_like(points[:, 0]) < fac
        points = points[mask]
    loss = V.squared_distance_to_edges(points).mean()
    if clamp:
        V.clamp()
    x = loss.item()
    loss.backward()
    optimizer.step()
    return x