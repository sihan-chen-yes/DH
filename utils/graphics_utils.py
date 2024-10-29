#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from typing import Union, Optional, Tuple, NamedTuple, Union
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
import cv2


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    tangents : np.array
    bitangents : np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def compute_tbn(geom, vt, vi, vti):
    """Computes tangent, bitangent, and normal vectors given a mesh.
    Args:
        geom: [N, n_verts, 3] torch.Tensor
        Vertex positions.
        vt: [n_uv_coords, 2] torch.Tensor
        UV coordinates.
        vi: [..., 3] torch.Tensor
        Face vertex indices.
        vti: [..., 3] torch.Tensor
        Face UV indices.
    Returns:
        [..., 3] torch.Tensors for T, B, N.
    """

    # v0 = geom[:, vi[..., 0]]
    # v1 = geom[:, vi[..., 1]]
    # v2 = geom[:, vi[..., 2]]

    v0 = geom[vi[..., 0]]
    v1 = geom[vi[..., 1]]
    v2 = geom[vi[..., 2]]
    vt0 = vt[vti[..., 0]]
    vt1 = vt[vti[..., 1]]
    vt2 = vt[vti[..., 2]]

    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0
    f = 1.0 / (
        vt01[None, ..., 0] * vt02[None, ..., 1]
        - vt01[None, ..., 1] * vt02[None, ..., 0]
    )
    tangent = f[..., None] * torch.stack(
        [
            v01[..., 0] * vt02[None, ..., 1] - v02[..., 0] * vt01[None, ..., 1],
            v01[..., 1] * vt02[None, ..., 1] - v02[..., 1] * vt01[None, ..., 1],
            v01[..., 2] * vt02[None, ..., 1] - v02[..., 2] * vt01[None, ..., 1],
        ],
        dim=-1,
    )
    # TODO not perpendicular
    tangent = F.normalize(tangent, dim=-1).squeeze()
    normal = F.normalize(torch.cross(v01, v02, dim=-1), dim=-1)
    bitangent = F.normalize(torch.cross(tangent, normal, dim=-1), dim=-1)

    return tangent, bitangent, normal

def compute_per_face_TBN(vertices, uvs, faces, normals=None):
    tangent_list = []
    bitangent_list = []
    normal_list = []
    epsilon = 1e-8

    # iterate faces
    for face_index, face in enumerate(faces):
        v_indices = face['vertex_indices']
        uv_indices = face['uv_indices']

        # xyz and uv coords
        v0, v1, v2 = [vertices[i] for i in v_indices]
        uv0, uv1, uv2 = [uvs[i] for i in uv_indices]

        delta_pos1 = np.array(v1) - np.array(v0)
        delta_pos2 = np.array(v2) - np.array(v0)
        delta_uv1 = np.array(uv1) - np.array(uv0)
        delta_uv2 = np.array(uv2) - np.array(uv0)

        r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0])
        tangent = r * (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1])
        # bitangent = r * (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0])
        if normals is None:
            bitangent = r * (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0])
            normal = np.cross(tangent, bitangent)
            normal = normal / (np.linalg.norm(normal) + epsilon)
        else:
            # if given normal then just use
            normal = np.array(normals[face_index])

        # Gram-Schmidt orthogonalization
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = tangent / np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)

        tangent_list.append(tangent.tolist())
        bitangent_list.append(bitangent.tolist())
        normal_list.append(normal.tolist())

    # change list into numpy
    return np.array(tangent_list), np.array(bitangent_list), np.array(normal_list)

def compute_per_vertex_TBN(vertex_neighbors, per_face_tangents, per_face_bitangents, per_face_normals):
    num_vertices = len(vertex_neighbors)
    per_vertex_tangents = np.zeros((num_vertices, 3))
    per_vertex_bitangents = np.zeros((num_vertices, 3))
    per_vertex_normals = np.zeros((num_vertices, 3))

    epsilon = 1e-8

    for vertex_idx, neighbors in enumerate(vertex_neighbors):
        tangent_sum = np.zeros(3)
        bitangent_sum = np.zeros(3)
        normal_sum = np.zeros(3)

        for neighbor in neighbors:
            tangent_sum += per_face_tangents[neighbor]
            bitangent_sum += per_face_bitangents[neighbor]
            normal_sum += per_face_normals[neighbor]

        # tangent_sum /= len(neighbors)
        # bitangent_sum /= len(neighbors)
        # normal_sum /= len(neighbors)

        # Normalizing TBN vectors
        per_vertex_tangents[vertex_idx] = tangent_sum / (np.linalg.norm(tangent_sum) + epsilon)
        per_vertex_bitangents[vertex_idx] = bitangent_sum / (np.linalg.norm(bitangent_sum) + epsilon)
        per_vertex_normals[vertex_idx] = normal_sum / (np.linalg.norm(normal_sum) + epsilon)

    return per_vertex_tangents, per_vertex_bitangents, per_vertex_normals

def get_TBN_map(uvs, faces, uv_mask, geo_module, per_vertex_tangents, per_vertex_bitangents, per_vertex_normals):
    # Create an empty image, map [0,1]x[0,1] to [0,1024]x[0,1024]
    # img_size = 1024
    # # interpolation pts in each bounding box of a triangle
    # interpolation_pts = 30
    # T_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # B_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # N_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    #
    # # Iterate through each face and draw TBN as colors
    # for face in faces:
    #     v_indices = face['vertex_indices']
    #     uv_indices = face['uv_indices']
    #     uv_coords = [uvs[uv_idx] for uv_idx in uv_indices]
    #     uv0, uv1, uv2 = uv_coords
    #     # S = det|a b| / 2
    #     # careful with clock wise
    #     area = 0.5 * ((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1]))
    #     # S = cross(a, b) / 2 very slow !
    #     # area = get_triangle_area(uv0, uv1, uv2)
    #
    #     # Compute the bounding box for the face in UV space
    #     min_u = min(uv[0] for uv in uv_coords)
    #     max_u = max(uv[0] for uv in uv_coords)
    #     min_v = min(uv[1] for uv in uv_coords)
    #     max_v = max(uv[1] for uv in uv_coords)
    #
    #     # Iterate through the pixels within the bounding box
    #     for u in np.linspace(min_u, max_u, num=interpolation_pts):
    #         for v in np.linspace(min_v, max_v, num=interpolation_pts):
    #             # Barycentric interpolation to check if the point is inside the triangle
    #             w0 = ((uv1[0] - u) * (uv2[1] - v) - (uv2[0] - u) * (uv1[1]- v)) * 0.5 / area
    #             # w0 = get_triangle_area([u, v], uv1, uv2) / area
    #             w1 = ((uv2[0] - u) * (uv0[1] - v) - (uv0[0] - u) * (uv2[1] - v)) * 0.5 / area
    #             # w1 = get_triangle_area([u, v], uv2, uv0) / area
    #             w2 = 1 - w0 - w1
    #
    #             # if inside the target UV space triangle
    #             if w0 >= 0 and w1 >= 0 and w2 >= 0:
    #                 # Interpolate TBN using barycentric coordinates
    #                 tangent = w0 * per_vertex_tangents[v_indices[0]] + \
    #                           w1 * per_vertex_tangents[v_indices[1]] + \
    #                           w2 * per_vertex_tangents[v_indices[2]]
    #                 bitangent = w0 * per_vertex_bitangents[v_indices[0]] + \
    #                             w1 * per_vertex_bitangents[v_indices[1]] + \
    #                             w2 * per_vertex_bitangents[v_indices[2]]
    #                 normal = w0 * per_vertex_normals[v_indices[0]] + \
    #                          w1 * per_vertex_normals[v_indices[1]] + \
    #                          w2 * per_vertex_normals[v_indices[2]]
    #
    #                 # Convert UV coordinates to image space
    #                 x = int(u * (img_size - 1))
    #                 y = int((1 - v) * (img_size - 1))
    #
    #                 # Normalize TBN vectors to [0, 255] for visualization
    #                 # map [-1, 1] to [0, 255]
    #                 tangent_color = ((tangent + 1) * 0.5 * 255).astype(np.uint8)
    #                 bitangent_color = ((bitangent + 1) * 0.5 * 255).astype(np.uint8)
    #                 normal_color = ((normal + 1) * 0.5 * 255).astype(np.uint8)
    #
    #                 # Draw the colors at the corresponding UV location
    #                 T_img[y, x] = tangent_color
    #                 B_img[y, x] = bitangent_color
    #                 N_img[y, x] = normal_color

    T_img = ((geo_module.to_uv(torch.tensor(per_vertex_tangents)) + 1) * 0.5 * 255).to(torch.uint8)
    B_img = ((geo_module.to_uv(torch.tensor(per_vertex_bitangents)) + 1) * 0.5 * 255).to(torch.uint8)
    N_img = ((geo_module.to_uv(torch.tensor(per_vertex_normals)) + 1) * 0.5 * 255).to(torch.uint8)

    T_img = np.array(T_img)
    T_img = cv2.flip(T_img, 0)
    T_img[~uv_mask] = [0, 0, 0]

    B_img = np.array(B_img)
    B_img = cv2.flip(B_img, 0)
    B_img[~uv_mask] = [0, 0, 0]

    N_img = np.array(N_img)
    N_img = cv2.flip(N_img, 0)
    N_img[~uv_mask] = [0, 0, 0]

    return T_img, B_img, N_img

def get_triangle_area(v0, v1, v2):
    v0_np = np.array(v0)
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    v0v1 = v1_np - v0_np
    v0v2 = v2_np - v0_np
    # prevent wise args not following clockwise
    return 0.5 * abs(np.cross(v0v1, v0v2))

def read_obj(obj_path):
    """
    Parameters
    ----------
    obj_path

    Returns
    -------
    vertices_uv
    faces: contains both xyz vertices index and uv vertices index
    vertex_neighbors: adjacent face index
    """
    # careful these xyz positions are not canonical pose xyz positions!
    vertices_xyz = []
    vertices_uv = []
    faces = []
    vertex_neighbors = []
    # vertex_indices
    vi = []
    # uv_indices
    vti = []

    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices_xyz.append([float(x), float(y), float(z)])
            elif line.startswith('vt '):
                _, u, v = line.strip().split()
                vertices_uv.append([float(u), float(v)])
            elif line.startswith('f '):
                face_elements = line.strip().split()[1:]
                vertex_indices = []
                uv_indices = []
                for element in face_elements:
                    parts = element.split('/')
                    vertex_indices.append(int(parts[0]) - 1)
                    if len(parts) > 1 and parts[1]:
                        uv_indices.append(int(parts[1]) - 1)
                faces.append({'vertex_indices': vertex_indices, 'uv_indices': uv_indices})
                vi.append(vertex_indices)
                vti.append(uv_indices)
    for i in range(len(vertices_xyz)):
        vertex_neighbor = []
        # iterate all faces
        for j, item in enumerate(faces):
            # contained in xyz face
            if i in item['vertex_indices']:
                vertex_neighbor.append(j)
        vertex_neighbors.append(vertex_neighbor)

    return {
        "vertices_uv": vertices_uv,
        "faces": faces,
        "vertex_neighbors": vertex_neighbors,
        "vi": vi,
        "vti": vti,
    }

class GeometryModule(torch.nn.Module):
    def __init__(
        self,
        vi,
        vt,
        vti,
        v2uv,
        uv_size=1024,
        flip_uv=False,
        impaint=False,
        impaint_threshold=100.0,
    ):
        super().__init__()

        self.register_buffer("vi", torch.as_tensor(vi))
        self.register_buffer("vt", torch.as_tensor(vt))
        self.register_buffer("vti", torch.as_tensor(vti))
        self.register_buffer("v2uv", torch.as_tensor(v2uv, dtype=torch.int64))

        # TODO: should we just pass topology here?
        self.n_verts = v2uv.shape[0]

        self.uv_size = uv_size

        # TODO: can't we just index face_index?
        index_image = make_uv_vert_index(
            self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        ).cpu()
        face_index, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        if impaint:
            if uv_size >= 1024:
                # logger.info(
                #     "impainting index image might take a while for sizes >= 1024"
                # )
                print("impainting index image might take a while for sizes >= 1024")

            index_image, bary_image = index_image_impaint(
                index_image, bary_image, impaint_threshold
            )
            # TODO: we can avoid doing this 2x
            face_index = index_image_impaint(
                face_index, distance_threshold=impaint_threshold
            )

        self.register_buffer("index_image", index_image.cpu())
        self.register_buffer("bary_image", bary_image.cpu())
        self.register_buffer("face_index_image", face_index.cpu())

    def render_index_images(self, uv_size, flip_uv=False, impaint=False):
        index_image = make_uv_vert_index(
            self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        face_image, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )

        if impaint:
            index_image, bary_image = index_image_impaint(
                index_image,
                bary_image,
            )

        return index_image, face_image, bary_image

    def vn(self, verts):
        return vert_normals(verts, self.vi[np.newaxis].to(torch.long))

    def to_uv(self, values):
        return values_to_uv(values, self.index_image, self.bary_image)

    def from_uv(self, values_uv):
        # TODO: we need to sample this
        return sample_uv(values_uv, self.vt, self.v2uv.to(torch.long))

def sample_uv(
    values_uv,
    uv_coords,
    v2uv: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
    flip_uvs: bool = False,
):
    batch_size = values_uv.shape[0]

    if flip_uvs:
        uv_coords = uv_coords.clone()
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]

    uv_coords_norm = (uv_coords * 2.0 - 1.0)[np.newaxis, :, np.newaxis].expand(
        batch_size, -1, -1, -1
    )
    values = (
        F.grid_sample(values_uv, uv_coords_norm, align_corners=align_corners, mode=mode)
        .squeeze(-1)
        .permute((0, 2, 1))
    )

    if v2uv is not None:
        values_duplicate = values[:, v2uv]
        values = values_duplicate.mean(2)

    return values


def values_to_uv(values, index_img, bary_img):
    """

    Parameters
    ----------
    values [n_verts, ...]
    index_img [uv_size, uv_size, 3]
    bary_img [uv_size, uv_size, 3]

    Returns
    -------
    values_uv
    [uv_size, uv_size, 3]

    """
    uv_size = index_img.shape[0]
    index_mask = torch.all(index_img != -1, dim=-1)
    idxs_flat = index_img[index_mask].to(torch.int64)
    bary_flat = bary_img[index_mask].to(torch.float32)
    # NOTE: here we assume
    values_flat = torch.sum(values[idxs_flat] * bary_flat[..., None], dim=1)
    values_uv = values_flat.reshape(uv_size, uv_size, -1)
    # values_uv[:, index_mask] = values_flat
    return values_uv


def face_normals(v, vi, eps: float = 1e-5):
    pts = v[:, vi]
    v0 = pts[:, :, 1] - pts[:, :, 0]
    v1 = pts[:, :, 2] - pts[:, :, 0]
    n = torch.cross(v0, v1, dim=-1)
    norm = torch.norm(n, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    n /= norm
    return n


def vert_normals(v, vi, eps: float = 1.0e-5):
    fnorms = face_normals(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(1, -1).expand(v.shape[0], -1)
    vnorms = torch.zeros_like(v)
    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = torch.norm(vnorms, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    vnorms /= norm
    return vnorms


def compute_view_cos(verts, faces, camera_pos):
    vn = F.normalize(vert_normals(verts, faces), dim=-1)
    v2c = F.normalize(verts - camera_pos[:, np.newaxis], dim=-1)
    return torch.einsum("bnd,bnd->bn", vn, v2c)

def compute_v2uv(n_verts, vi, vti, n_max=4):
    """Computes mapping from vertex indices to texture indices.

    Args:
        vi: [F, 3], triangles
        vti: [F, 3], texture triangles
        n_max: int, max number of texture locations

    Returns:
        [n_verts, n_max], texture indices
    """
    v2uv_dict = {}
    for i_v, i_uv in zip(vi.reshape(-1), vti.reshape(-1)):
        v2uv_dict.setdefault(i_v, set()).add(i_uv)
    assert len(v2uv_dict) == n_verts
    v2uv = np.zeros((n_verts, n_max), dtype=np.int32)
    for i in range(n_verts):
        vals = sorted(list(v2uv_dict[i]))
        v2uv[i, :] = vals[0]
        v2uv[i, : len(vals)] = np.array(vals)
    return v2uv


def compute_neighbours(n_verts, vi, n_max_values=10):
    """Computes first-ring neighbours given vertices and faces."""
    n_vi = vi.shape[0]

    adj = {i: set() for i in range(n_verts)}
    for i in range(n_vi):
        for idx in vi[i]:
            adj[idx] |= set(vi[i]) - set([idx])

    nbs_idxs = np.tile(np.arange(n_verts)[:, np.newaxis], (1, n_max_values))
    nbs_weights = np.zeros((n_verts, n_max_values), dtype=np.float32)

    for idx in range(n_verts):
        n_values = min(len(adj[idx]), n_max_values)
        nbs_idxs[idx, :n_values] = np.array(list(adj[idx]))[:n_values]
        nbs_weights[idx, :n_values] = -1.0 / n_values

    return nbs_idxs, nbs_weights


def make_postex(v, idxim, barim):
    return (
        barim[None, :, :, 0, None] * v[:, idxim[:, :, 0]]
        + barim[None, :, :, 1, None] * v[:, idxim[:, :, 1]]
        + barim[None, :, :, 2, None] * v[:, idxim[:, :, 2]]
    ).permute(0, 3, 1, 2)


def matrix_to_axisangle(r):
    th = torch.arccos(0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0))[..., None]
    vec = (
        0.5
        * torch.stack(
            [
                r[..., 2, 1] - r[..., 1, 2],
                r[..., 0, 2] - r[..., 2, 0],
                r[..., 1, 0] - r[..., 0, 1],
            ],
            dim=-1,
        )
        / torch.sin(th)
    )
    return th, vec


def axisangle_to_matrix(rvec):
    theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=-1))
    rvec = rvec / theta[..., None]
    costh = torch.cos(theta)
    sinth = torch.sin(theta)
    return torch.stack(
        (
            torch.stack(
                (
                    rvec[..., 0] ** 2 + (1.0 - rvec[..., 0] ** 2) * costh,
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) - rvec[..., 2] * sinth,
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) + rvec[..., 1] * sinth,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) + rvec[..., 2] * sinth,
                    rvec[..., 1] ** 2 + (1.0 - rvec[..., 1] ** 2) * costh,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) - rvec[..., 0] * sinth,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) - rvec[..., 1] * sinth,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) + rvec[..., 0] * sinth,
                    rvec[..., 2] ** 2 + (1.0 - rvec[..., 2] ** 2) * costh,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def rotation_interp(r0, r1, alpha):
    r0a = r0.view(-1, 3, 3)
    r1a = r1.view(-1, 3, 3)
    r = torch.bmm(r0a.permute(0, 2, 1), r1a).view_as(r0)

    th, rvec = matrix_to_axisangle(r)
    rvec = rvec * (alpha * th)

    r = axisangle_to_matrix(rvec)
    return torch.bmm(r0a, r.view(-1, 3, 3)).view_as(r0)


def convert_camera_parameters(Rt, K):
    R = Rt[:, :3, :3]
    t = -R.permute(0, 2, 1).bmm(Rt[:, :3, 3].unsqueeze(2)).squeeze(2)
    return dict(
        campos=t,
        camrot=R,
        focal=K[:, :2, :2],
        princpt=K[:, :2, 2],
    )


def project_points_multi(p, Rt, K, normalize=False, size=None):
    """Project a set of 3D points into multiple cameras with a pinhole model.
    Args:
        p: [B, N, 3], input 3D points in world coordinates
        Rt: [B, NC, 3, 4], extrinsics (where NC is the number of cameras to project to)
        K: [B, NC, 3, 3], intrinsics
        normalize: bool, whether to normalize coordinates to [-1.0, 1.0]
    Returns:
        tuple:
        - [B, NC, N, 2] - projected points
        - [B, NC, N] - their
    """
    B, N = p.shape[:2]
    NC = Rt.shape[1]

    Rt = Rt.reshape(B * NC, 3, 4)
    K = K.reshape(B * NC, 3, 3)

    # [B, N, 3] -> [B * NC, N, 3]
    p = p[:, np.newaxis].expand(-1, NC, -1, -1).reshape(B * NC, -1, 3)
    p_cam = p @ Rt[:, :3, :3].mT + Rt[:, :3, 3][:, np.newaxis]
    p_pix = p_cam @ K.mT
    p_depth = p_pix[:, :, 2:]
    p_pix = (p_pix[..., :2] / p_depth).reshape(B, NC, N, 2)
    p_depth = p_depth.reshape(B, NC, N)

    if normalize:
        assert size is not None
        h, w = size
        p_pix = (
            2.0 * p_pix / torch.as_tensor([w, h], dtype=torch.float32, device=p.device) - 1.0
        )
    return p_pix, p_depth

def xyz2normals(xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert XYZ image to normal image

    Args:
        xyz: torch.Tensor
        [B, 3, H, W] XYZ image

    Returns:
        torch.Tensor: [B, 3, H, W] image of normals
    """

    nrml = torch.zeros_like(xyz)
    xyz = torch.cat((xyz[:, :, :1, :] * 0, xyz[:, :, :, :], xyz[:, :, :1, :] * 0), dim=2)
    xyz = torch.cat((xyz[:, :, :, :1] * 0, xyz[:, :, :, :], xyz[:, :, :, :1] * 0), dim=3)
    U = (xyz[:, :, 2:, 1:-1] - xyz[:, :, :-2, 1:-1]) / -2
    V = (xyz[:, :, 1:-1, 2:] - xyz[:, :, 1:-1, :-2]) / -2

    nrml[:, 0, ...] = U[:, 1, ...] * V[:, 2, ...] - U[:, 2, ...] * V[:, 1, ...]
    nrml[:, 1, ...] = U[:, 2, ...] * V[:, 0, ...] - U[:, 0, ...] * V[:, 2, ...]
    nrml[:, 2, ...] = U[:, 0, ...] * V[:, 1, ...] - U[:, 1, ...] * V[:, 0, ...]
    veclen = torch.norm(nrml, dim=1, keepdim=True).clamp(min=eps)
    return nrml / veclen


# pyre-fixme[2]: Parameter must be annotated.
def depth2xyz(depth, focal, princpt) -> torch.Tensor:
    """Convert depth image to XYZ image using camera intrinsics

    Args:
        depth: torch.Tensor
        [B, 1, H, W] depth image

        focal: torch.Tensor
        [B, 2, 2] camera focal lengths

        princpt: torch.Tensor
        [B, 2] camera principal points

    Returns:
        torch.Tensor: [B, 3, H, W] XYZ image
    """

    b, h, w = depth.shape[0], depth.shape[2], depth.shape[3]
    ix = (
        torch.arange(w, device=depth.device).float()[None, None, :] - princpt[:, None, None, 0]
    ) / focal[:, None, None, 0, 0]
    iy = (
        torch.arange(h, device=depth.device).float()[None, :, None] - princpt[:, None, None, 1]
    ) / focal[:, None, None, 1, 1]
    xyz = torch.zeros((b, 3, h, w), device=depth.device)
    xyz[:, 0, ...] = depth[:, 0, :, :] * ix
    xyz[:, 1, ...] = depth[:, 0, :, :] * iy
    xyz[:, 2, ...] = depth[:, 0, :, :]
    return xyz


# pyre-fixme[2]: Parameter must be annotated.
def depth2normals(depth, focal, princpt) -> torch.Tensor:
    """Convert depth image to normal image using camera intrinsics

    Args:
        depth: torch.Tensor
        [B, 1, H, W] depth image

        focal: torch.Tensor
        [B, 2, 2] camera focal lengths

        princpt: torch.Tensor
        [B, 2] camera principal points

    Returns:
        torch.Tensor: [B, 3, H, W] normal image
    """

    return xyz2normals(depth2xyz(depth, focal, princpt))


def depth_discontuity_mask(
    depth: torch.Tensor, threshold: float = 40.0, kscale: float = 4.0, pool_ksize: int = 3
) -> torch.Tensor:
    device = depth.device

    with torch.no_grad():
        # TODO: pass the kernel?
        kernel = torch.as_tensor(
            [
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            ],
            dtype=torch.float32,
            device=device,
        )

        disc_mask = (torch.norm(F.conv2d(depth, kernel, bias=None, padding=1), dim=1) > threshold)[
            :, np.newaxis
        ]
        disc_mask = (
            F.avg_pool2d(disc_mask.float(), pool_ksize, stride=1, padding=pool_ksize // 2) > 0.0
        )

    return disc_mask

def make_uv_face_index(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    device: Optional[Union[str, torch.device]] = None,
):
    """Compute a UV-space face index map identifying which mesh face contains each
    texel. For texels with no assigned triangle, the index will be -1."""

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if device is not None:
        if isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device
        assert dev.type == "cuda"
    else:
        dev = torch.device("cuda")

    vt = 1.0 - vt.clone()

    # if flip_uv:
    #     vt = vt.clone()
    #     vt[:, 1] = 1 - vt[:, 1]
    vt_pix = 2.0 * vt.to(dev) - 1.0
    vt_pix = torch.cat([vt_pix, torch.ones_like(vt_pix[:, 0:1])], dim=1)
    meshes = Meshes(vt_pix[np.newaxis], vti[np.newaxis].to(dev))
    with torch.no_grad():
        face_index, _, _, _ = rasterize_meshes(
            meshes, uv_shape, faces_per_pixel=1, z_clip_value=0.0, bin_size=0
        )
        face_index = face_index[0, ..., 0]
    return face_index


def make_uv_vert_index(
    vt: torch.Tensor,
    vi: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space vertex index map identifying which mesh vertices
    comprise the triangle containing each texel. For texels with no assigned
    triangle, all indices will be -1.
    """
    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv).to(vi.device)
    vert_index_map = vi[face_index_map.clamp(min=0)]
    vert_index_map[face_index_map < 0] = -1
    return vert_index_map.long()


def bary_coords(points: torch.Tensor, triangles: torch.Tensor, eps: float = 1.0e-6):
    """Computes barycentric coordinates for a set of 2D query points given
    coordintes for the 3 vertices of the enclosing triangle for each point."""
    x = points[:, 0] - triangles[2, :, 0]
    x1 = triangles[0, :, 0] - triangles[2, :, 0]
    x2 = triangles[1, :, 0] - triangles[2, :, 0]
    y = points[:, 1] - triangles[2, :, 1]
    y1 = triangles[0, :, 1] - triangles[2, :, 1]
    y2 = triangles[1, :, 1] - triangles[2, :, 1]
    denom = y2 * x1 - y1 * x2
    n0 = y2 * x - x2 * y
    n1 = x1 * y - y1 * x

    # Small epsilon to prevent divide-by-zero error.
    denom = torch.where(denom >= 0, denom.clamp(min=eps), denom.clamp(max=-eps))

    bary_0 = n0 / denom
    bary_1 = n1 / denom
    bary_2 = 1.0 - bary_0 - bary_1

    return torch.stack((bary_0, bary_1, bary_2))


def make_uv_barys(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for that texel within its enclosing UV triangle. For texels
    with no assigned triangle, all 3 barycentric coordinates will be 0.
    """
    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv=False).to(vt.device)
    vti_map = vti.long()[face_index_map.clamp(min=0)]
    uv_tri_uvs = vt[vti_map].permute(2, 0, 1, 3)

    uv_grid = torch.meshgrid(
        torch.linspace(0.5, uv_shape[0] - 0.5, uv_shape[0]) / uv_shape[0],
        torch.linspace(0.5, uv_shape[1] - 0.5, uv_shape[1]) / uv_shape[1],
    )
    uv_grid = torch.stack(uv_grid[::-1], dim=2).to(uv_tri_uvs)

    bary_map = bary_coords(uv_grid.view(-1, 2), uv_tri_uvs.view(3, -1, 2))
    bary_map = bary_map.permute(1, 0).view(uv_shape[0], uv_shape[1], 3)
    bary_map[face_index_map < 0] = 0
    return face_index_map, bary_map

def index_image_impaint(
    index_image: torch.Tensor,
    bary_image: Optional[torch.Tensor] = None,
    distance_threshold=100.0,
):
    # getting the mask around the indexes?
    if len(index_image.shape) == 3:
        valid_index = (index_image != -1).any(dim=-1)
    elif len(index_image.shape) == 2:
        valid_index = index_image != -1
    else:
        raise ValueError("`index_image` should be a [H,W] or [H,W,C] image")

    invalid_index = ~valid_index

    device = index_image.device

    valid_ij = torch.stack(torch.where(valid_index), dim=-1)
    invalid_ij = torch.stack(torch.where(invalid_index), dim=-1)
    lookup_valid = KDTree(valid_ij.cpu().numpy())

    dists, idxs = lookup_valid.query(invalid_ij.cpu())

    # TODO: try average?
    idxs = torch.as_tensor(idxs, device=device)[..., 0]
    dists = torch.as_tensor(dists, device=device)[..., 0]

    dist_mask = dists < distance_threshold

    invalid_border = torch.zeros_like(invalid_index)
    invalid_border[invalid_index] = dist_mask

    invalid_src_ij = valid_ij[idxs][dist_mask]
    invalid_dst_ij = invalid_ij[dist_mask]

    index_image_imp = index_image.clone()

    index_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = index_image[
        invalid_src_ij[:, 0], invalid_src_ij[:, 1]
    ]

    if bary_image is not None:
        bary_image_imp = bary_image.clone()

        bary_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = bary_image[
            invalid_src_ij[:, 0], invalid_src_ij[:, 1]
        ]

        return index_image_imp, bary_image_imp
    return index_image_imp