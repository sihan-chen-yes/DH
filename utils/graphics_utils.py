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
from typing import NamedTuple

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

def compute_tangent_bitangent(vertices, uvs, faces, normals):
    tangents = []
    bitangents = []

    # iterate faces
    for face in faces:
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

        normal = np.array(normals[v_indices[0]])

        # Gram-Schmidt orthogonalization
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = tangent / np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)

        tangents.append(tangent.tolist())
        bitangents.append(bitangent.tolist())

    # change list into numpy
    return np.array(tangents), np.array(bitangents)