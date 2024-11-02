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

import os
import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset
from utils.graphics_utils import get_TBN_map
from utils.general_utils import build_rotation
import cv2
from pytorch3d.ops.knn import knn_points

class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        elif cfg.mode == 'reconstruct':
            self.test_dataset = load_dataset(cfg.dataset, split='reconstruct')
        else:
            raise ValueError


        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.save_mesh_TBN_map()
        self.save_gaussian_TBN_map()

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)

    def save_mesh_TBN_map(self):
        """
        save TBN map of mesh vertex TBN

        """
        map_dir = os.path.join(self.save_dir, "TBN_map")
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        print("exporting mesh TBN UV map ...")
        uvs = self.metadata["vertices_uv"]
        faces = self.metadata["faces"]
        per_vertex_tangents = self.metadata["per_vertex_tangents"]
        per_vertex_bitangents = self.metadata["per_vertex_bitangents"]
        per_vertex_normals = self.metadata["per_vertex_normals"]
        geo_module = self.metadata["geo_module"]
        uv_mask = self.metadata["uv_mask"]
        T_img, B_img, N_img = get_TBN_map(uvs, faces, uv_mask, geo_module, per_vertex_tangents, per_vertex_bitangents, per_vertex_normals)
        # Save the image
        cv2.imwrite(os.path.join(map_dir, 'T_mesh_map.png'), T_img)
        print("exported mesh T UV map")
        cv2.imwrite(os.path.join(map_dir, 'B_mesh_map.png'), B_img)
        print("exported mesh B UV map")
        cv2.imwrite(os.path.join(map_dir, 'N_mesh_map.png'), N_img)
        print("exported mesh N UV map")

    def save_gaussian_TBN_map(self):
        """
        save TBN map of gaussian vertex TBN
        replace mesh TBN with neareast gaussian TBN
        """
        map_dir = os.path.join(self.save_dir, "TBN_map")
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        print("exporting gaussian TBN UV map ...")
        mesh_vertex = torch.tensor(self.metadata["vertices_xyz"]).float().cuda()
        gaussian_vertex = self.gaussians.get_xyz

        _, nn_ix, _ = knn_points(mesh_vertex.unsqueeze(0),
                                 gaussian_vertex.unsqueeze(0),
                                 K=1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0).squeeze(1)
        # remove gradient
        rotations = build_rotation(self.gaussians.get_rotation).detach()
        # per_vertex TBN w.r.t nearest gaussian TBN
        per_vertex_tangents = rotations[:, :, 0][nn_ix].cpu().numpy()
        per_vertex_bitangents = rotations[:, :, 1][nn_ix].cpu().numpy()
        per_vertex_normals = rotations[:, :, 2][nn_ix].cpu().numpy()

        uvs = self.metadata["vertices_uv"]
        faces = self.metadata["faces"]
        geo_module = self.metadata["geo_module"]
        uv_mask = self.metadata["uv_mask"]
        T_img, B_img, N_img = get_TBN_map(uvs, faces, uv_mask, geo_module, per_vertex_tangents, per_vertex_bitangents, per_vertex_normals)
        # Save the image
        cv2.imwrite(os.path.join(map_dir, 'T_gaussian_map.png'), T_img)
        print("exported gaussian T UV map")
        cv2.imwrite(os.path.join(map_dir, 'B_gaussian_map.png'), B_img)
        print("exported gaussian B UV map")
        cv2.imwrite(os.path.join(map_dir, 'N_gaussian_map.png'), N_img)
        print("exported gaussian N UV map")