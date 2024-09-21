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
import torchgeometry as tgm
from pytorch3d.ops.knn import knn_points

class Interpolater(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.mesh = trimesh.Trimesh(vertices=self.metadata['smpl_verts'], faces=self.metadata['faces'])

        # mesh vertex properties
        self._vertex_xyz = torch.tensor(self.mesh.vertices, dtype=torch.float, device="cuda")
        # TODO to use
        self._vertex_normal = torch.tensor(self.mesh.vertex_normals, dtype=torch.float, device="cuda")
        self._LBS_weight = torch.tensor(self.metadata['skinning_weights'], dtype=torch.float, device="cuda")
        # TODO better initialization
        self._vertex_scaling = nn.Parameter(torch.zeros((self._vertex_xyz.shape[0], 3), device="cuda"))
        self._vertex_rotation = nn.Parameter(torch.ones((self._vertex_xyz.shape[0], 4), device="cuda"))

        self.displacement_predictor = VanillaCondMLP(dim_in=3, dim_cond=24 * 9, dim_out=3, config=cfg.mlp)
        #TODO
        self.setup_functions()

        self.Kg = cfg.KNN.Kg
        self.Kv = cfg.KNN.Kv
        self.base_scale = cfg.base_scale

    def forward(self, gaussians, camera, compute_loss=True):
        interpolated_gaussians = gaussians.clone()

        rots = camera.rots
        # TODO
        # pose = tgm.rotation_matrix_to_angle_axis(rots.reshape(-1, 3, 3))
        pose = rots.flatten()
        displacement = self.displacement_predictor(self._vertex_xyz, pose)

        #interpolation
        xyz_gaussians = interpolated_gaussians.get_xyz

        # TODO fast KNN
        _, nn_ix, _ = knn_points(xyz_gaussians.unsqueeze(0),
                                 self._vertex_xyz.unsqueeze(0),
                                 K=self.Kg,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

        # nearest vertices
        nearest_xyz_vertices = self._vertex_xyz[nn_ix]
        # TODO
        interpolate_weights = 1 / (torch.norm((xyz_gaussians[:, None, :] - nearest_xyz_vertices), dim=2) + 1e-5)

        interpolated_gaussians._xyz = torch.bmm(interpolate_weights.unsqueeze(1), displacement[nn_ix]).squeeze(1) + xyz_gaussians
        interpolated_gaussians._scaling = torch.bmm(interpolate_weights.unsqueeze(1), self._vertex_scaling[nn_ix]).squeeze(1)
        interpolated_gaussians._rotation = torch.bmm(interpolate_weights.unsqueeze(1), self._vertex_rotation[nn_ix]).squeeze(1)

        if compute_loss:
            # regularization
            loss_mdist = torch.norm(xyz_gaussians.unsqueeze(1) - nearest_xyz_vertices, p=2, dim=2).mean()
            loss_disp = torch.norm(displacement, p=2, dim=1).mean()
            loss_base_scale = torch.clamp(interpolated_gaussians._scaling - self.base_scale, min=0).mean()
            loss_reg = {
                "mdist": loss_mdist,
                "disp": loss_disp,
                'base_scale': loss_base_scale
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