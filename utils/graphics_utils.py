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

def compute_per_face_TBN(vertices, uvs, faces, normals=None):
    tangent_list = []
    bitangent_list = []
    normal_list = []
    epsilon = 1e-8

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
        if normals is None:
            bitangent = r * (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0])
            normal = np.cross(tangent, bitangent)
            normal = normal / (np.linalg.norm(normal) + epsilon)
        else:
            # if given normal then just use
            normal = np.array(normals[v_indices[0]])

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

def get_TBN_map(uvs, faces, per_vertex_tangents, per_vertex_bitangents, per_vertex_normals):
    # Create an empty image, map [0,1]x[0,1] to [0,1024]x[0,1024]
    img_size = 1024
    # interpolation pts in each bounding box of a triangle
    interpolation_pts = 30
    T_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    B_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    N_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Iterate through each face and draw TBN as colors
    for face in faces:
        v_indices = face['vertex_indices']
        uv_indices = face['uv_indices']
        uv_coords = [uvs[uv_idx] for uv_idx in uv_indices]
        uv0, uv1, uv2 = uv_coords
        # determinant
        area = 0.5 * ((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1]))

        # Compute the bounding box for the face in UV space
        min_u = min(uv[0] for uv in uv_coords)
        max_u = max(uv[0] for uv in uv_coords)
        min_v = min(uv[1] for uv in uv_coords)
        max_v = max(uv[1] for uv in uv_coords)

        # Iterate through the pixels within the bounding box
        for u in np.linspace(min_u, max_u, num=interpolation_pts):
            for v in np.linspace(min_v, max_v, num=interpolation_pts):
                # Barycentric interpolation to check if the point is inside the triangle
                w0 = ((uv1[1] - uv2[1]) * (u - uv2[0]) - (uv1[0] - uv2[0]) * (v - uv2[1])) / (2 * area)
                w1 = ((uv2[1] - uv0[1]) * (u - uv2[0]) - (uv2[0] - uv0[0]) * (v - uv2[1])) / (2 * area)
                w2 = 1 - w0 - w1

                # if inside the target UV space triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate TBN using barycentric coordinates
                    tangent = w0 * per_vertex_tangents[v_indices[0]] + \
                              w1 * per_vertex_tangents[v_indices[1]] + \
                              w2 * per_vertex_tangents[v_indices[2]]
                    bitangent = w0 * per_vertex_bitangents[v_indices[0]] + \
                                w1 * per_vertex_bitangents[v_indices[1]] + \
                                w2 * per_vertex_bitangents[v_indices[2]]
                    normal = w0 * per_vertex_normals[v_indices[0]] + \
                             w1 * per_vertex_normals[v_indices[1]] + \
                             w2 * per_vertex_normals[v_indices[2]]

                    # Convert UV coordinates to image space
                    x = int(u * (img_size - 1))
                    y = int((1 - v) * (img_size - 1))

                    # Normalize TBN vectors to [0, 255] for visualization
                    # map [-1, 1] to [0, 255]
                    tangent_color = ((tangent + 1) * 0.5 * 255).astype(np.uint8)
                    bitangent_color = ((bitangent + 1) * 0.5 * 255).astype(np.uint8)
                    normal_color = ((normal + 1) * 0.5 * 255).astype(np.uint8)

                    # Draw the colors at the corresponding UV location
                    T_img[y, x] = tangent_color
                    B_img[y, x] = bitangent_color
                    N_img[y, x] = normal_color

    return T_img, B_img, N_img