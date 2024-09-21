import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from models.network_utils import VanillaCondMLP
import trimesh
import igl
from pytorch3d.transforms import matrix_to_euler_angles

from pytorch3d.ops.knn import knn_points

class Interpolater(nn.Module):

    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.mesh = trimesh.Trimesh(vertices=self.metadata['smpl_verts'], faces=self.metadata['faces'])

        # mesh vertex properties: fixed
        self._vertex_xyz = torch.tensor(self.mesh.vertices, dtype=torch.float, device="cuda")
        # TODO to use
        self._vertex_normal = torch.tensor(self.mesh.vertex_normals, dtype=torch.float, device="cuda")
        self._LBS_weight = torch.tensor(self.metadata['skinning_weights'], dtype=torch.float, device="cuda")

        dist2 = torch.clamp_min(distCUDA2(self._vertex_xyz).float(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        self._vertex_scaling = nn.Parameter(scales)
        self._vertex_rotation = nn.Parameter(torch.rand((self._vertex_xyz.shape[0], 4), device="cuda"))

        self.displacement_predictor = VanillaCondMLP(dim_in=3, dim_cond=24 * 3, dim_out=3, config=cfg.mlp)

        self.Kg = cfg.KNN.Kg
        self.Kv = cfg.KNN.Kv
        self.base_scale = cfg.base_scale

    def forward(self, gaussians, camera, compute_loss=True):
        interpolated_gaussians = gaussians.clone()

        rots = camera.rots
        pose = matrix_to_euler_angles(rots.reshape(-1, 3, 3), "ZYX")
        pose = pose.flatten()
        displacement = self.displacement_predictor(self._vertex_xyz, pose)

        #interpolation
        gaussians_xyz = interpolated_gaussians.get_xyz

        # TODO fast KNN
        _, nn_ix, _ = knn_points(gaussians_xyz.unsqueeze(0),
                                 self._vertex_xyz.unsqueeze(0),
                                 K=self.Kg,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

        # nearest vertices
        nearest_xyz_vertices = self._vertex_xyz[nn_ix]
        # TODO
        interpolate_weights = 1 / (torch.norm((gaussians_xyz[:, None, :] - nearest_xyz_vertices), dim=2, p=2) + 1e-7)
        interpolate_weights = interpolate_weights / (interpolate_weights.sum(dim=1, keepdim=True) + 1e-7)
        # change property space when interpolation
        interpolated_gaussians._xyz = torch.bmm(interpolate_weights.unsqueeze(1), displacement[nn_ix]).squeeze(1) + gaussians_xyz
        interpolated_gaussians._scaling = torch.log(torch.bmm(
            interpolate_weights.unsqueeze(1), torch.exp(self._vertex_scaling)[nn_ix]).squeeze(1))
        # TODO quaternion interpolation
        interpolated_gaussians._rotation = torch.bmm(
            interpolate_weights.unsqueeze(1), torch.nn.functional.normalize(self._vertex_rotation)[nn_ix]).squeeze(1)

        interpolated_gaussians.LBS_weight = torch.bmm(interpolate_weights.unsqueeze(1), self._LBS_weight[nn_ix]).squeeze(1)
        if compute_loss:
            # regularization
            loss_mdist = torch.norm(gaussians_xyz.unsqueeze(1) - nearest_xyz_vertices, p=2, dim=2).mean()
            loss_disp = torch.norm(displacement, p=2, dim=1).mean()
            loss_base_scale = torch.clamp(interpolated_gaussians.get_scaling - self.base_scale, min=0).mean()
            gs_normal = build_rotation(interpolated_gaussians._rotation)[:, :, -1]
            interpolated_normal = torch.bmm(interpolate_weights.unsqueeze(1), self._vertex_normal[nn_ix]).squeeze(1)
            loss_mesh_normal = torch.norm(gs_normal - interpolated_normal, p=2, dim=1).mean()
            loss_reg = {
                "mdist": loss_mdist,
                "disp": loss_disp,
                "base_scale": loss_base_scale,
                "mesh_normal": loss_mesh_normal,
            }
        else:
            loss_reg = {}

        return interpolated_gaussians, loss_reg

def get_interpolater(cfg, metadata):
    name = cfg.name
    model_dict = {
        "mlp": Interpolater,
    }
    return model_dict[name](cfg, metadata)