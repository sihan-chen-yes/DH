import torch
import torch.nn as nn

from utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from utils.general_utils import build_rotation, linear_to_srgb
from models.network_utils import VanillaCondMLP

class ColorPrecompute(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

    def forward(self, gaussians, camera):
        raise NotImplementedError

class SH2RGB(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        self.use_ref = cfg.get('use_ref', False)

    def forward(self, gaussians, camera):
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        # normalize
        dir_pp = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        # use reflection direction of view direction w.r.t normal
        if self.use_ref:
            # normalized via activation
            normal = build_rotation(gaussians.get_rotation)[:, :, -1]
            # incident ray
            w_i = -dir_pp
            # reflection ray, normalized
            w_r = 2 * (torch.bmm(w_i.unsqueeze(1), normal.unsqueeze(2)).squeeze(2)) * normal - w_i
            dir_pp = w_r
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.get('view_noise', 0.)
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp
        
class ColorMLP(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        d_in = cfg.feature_dim

        self.use_xyz = cfg.get('use_xyz', False)
        self.use_cov = cfg.get('use_cov', False)
        self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.latent_dim = cfg.get('latent_dim', 0)

        self.use_ref = cfg.get('use_ref', False)

        if self.use_xyz:
            d_in += 3
        if self.use_cov:
            d_in += 6 # only upper triangle suffice
        if self.use_normal:
            d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim
        if self.latent_dim > 0:
            d_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3
        self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)
        self.color_activation = nn.Sigmoid()

    def compose_input(self, gaussians, camera):
        features = gaussians.get_features.squeeze(-1)
        n_points = features.shape[0]
        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            features = torch.cat([features, xyz_norm], dim=1)
        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)
        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
            features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            # normalize
            dir_pp = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            # use reflection direction of view direction w.r.t normal
            if self.use_ref:
                # normalized via activation
                normal = build_rotation(gaussians.get_rotation)[:, :, -1]
                # incident ray
                w_i = -dir_pp
                # reflection ray, normalized
                w_r = 2 * (torch.bmm(w_i.unsqueeze(1), normal.unsqueeze(2)).squeeze(2)) * normal - w_i
                dir_pp = w_r
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.get('view_noise', 0.)
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_embed = self.sh_embed(dir_pp)
            features = torch.cat([features, dir_embed], dim=1)
        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(features.shape[0], -1)
            features = torch.cat([features, latent_code], dim=1)

        return features


    def forward(self, gaussians, camera):
        inp = self.compose_input(gaussians, camera)
        output = self.mlp(inp)
        color = self.color_activation(output)
        return color

class FusionMLP(ColorPrecompute):
    '''
    3 MLPs:
    diffuse_mlp(gs_features, latent_features),
    specular_mlp(reflection_direction_features, latent_features),
    blending_mlp(gs_features, latent_features, geometry_normal_features)
    shading_normal_mlp(gs_features, latent_features, geometry_normal_features)
    '''
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        d_diffuse_in = cfg.feature_dim
        d_specular_in = 0
        d_blending_in = cfg.feature_dim

        d_shading_normal_in = cfg.feature_dim
        # self.use_xyz = cfg.get('use_xyz', False)
        # self.use_cov = cfg.get('use_cov', False)
        # self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        # self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.latent_dim = cfg.get('latent_dim', 0)

        self.use_ref = cfg.get('use_ref', False)
        self.texture_mode = cfg.get('texture_mode', 'fusion')

        # if self.use_xyz:
        #     d_in += 3
        # if self.use_cov:
        #     d_in += 6 # only upper triangle suffice
        # if self.use_normal:
        #     d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_specular_in += (self.sh_degree + 1) ** 2 - 1
            d_blending_in += (self.sh_degree + 1) ** 2 - 1
            d_shading_normal_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        # if self.non_rigid_dim > 0:
        #     d_in += self.non_rigid_dim

        # TODO
        if self.latent_dim > 0:
            d_diffuse_in += self.latent_dim
            d_specular_in += self.latent_dim
            d_blending_in += self.latent_dim
            d_shading_normal_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_color_out = 3
        d_blending_out = 1
        d_shading_normal_out = 3
        self.diffuse_mlp = VanillaCondMLP(d_diffuse_in, 0, d_color_out, cfg.mlp)
        self.specular_mlp = VanillaCondMLP(d_specular_in, 0, d_color_out, cfg.mlp)
        self.blending_mlp = VanillaCondMLP(d_blending_in, 0, d_blending_out, cfg.mlp)
        self.shading_normal_mlp = VanillaCondMLP(d_shading_normal_in, 0, d_shading_normal_out, cfg.mlp)
        self.color_activation = nn.Sigmoid()
        self.blending_activation = nn.Sigmoid()
        self.shading_normal_offset_activation = nn.Tanh()
        self.normal_activation = torch.nn.functional.normalize

    def compose_shading_input(self, gaussians, camera):
        shading_normal_features = gaussians.get_features.squeeze(-1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(shading_normal_features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(shading_normal_features.shape[0], -1)
            shading_normal_features = torch.cat([shading_normal_features, latent_code], dim=1)

        normal = self.normal_activation(build_rotation(gaussians.get_rotation)[:, :, -1])
        if self.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            normal = torch.matmul(R_bwd, normal.unsqueeze(-1)).squeeze(-1)
            # view_noise_scale = self.cfg.get('view_noise', 0.)
            # if self.training and view_noise_scale > 0.:
            #     view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
            #                               dtype=torch.float32,
            #                               device=normal.device).transpose(0, 1)
            #     normal = torch.matmul(normal, view_noise)
        normal_embed = self.sh_embed(normal)
        shading_normal_features = torch.cat([shading_normal_features, normal_embed], dim=1)
        return shading_normal_features

    def compose_input(self, gaussians, camera, shading_normal_offset=None):
        diffuse_features = gaussians.get_features.squeeze(-1)
        specular_features = torch.empty(0).cuda()
        blending_features = gaussians.get_features.squeeze(-1)
        n_points = diffuse_features.shape[0]
        if shading_normal_offset != None:
            normal = self.normal_activation(build_rotation(gaussians.get_rotation)[:, :, -1] + shading_normal_offset)
        else:
            normal = self.normal_activation(build_rotation(gaussians.get_rotation)[:, :, -1])
        # if self.use_xyz:
        #     aabb = self.metadata["aabb"]
        #     xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
        #     features = torch.cat([features, xyz_norm], dim=1)
        # if self.use_cov:
        #     cov = gaussians.get_covariance()
        #     features = torch.cat([features, cov], dim=1)
        # if self.use_normal:
        #     scale = gaussians._scaling
        #     rot = build_rotation(gaussians._rotation)
        #     normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        #     features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            # normalize
            dir_pp = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            # use reflection direction of view direction w.r.t normal
            if self.use_ref:
                # incident ray
                w_i = -dir_pp
                # reflection ray, normalized
                w_r = 2 * (torch.bmm(w_i.unsqueeze(1), normal.unsqueeze(2)).squeeze(2)) * normal - w_i
                dir_pp = w_r
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.get('view_noise', 0.)
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_embed = self.sh_embed(dir_pp)
            specular_features = torch.cat([specular_features, dir_embed], dim=1)
        # if self.non_rigid_dim > 0:
        #     assert hasattr(gaussians, "non_rigid_feature")
        #     features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(diffuse_features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(diffuse_features.shape[0], -1)
            diffuse_features = torch.cat([diffuse_features, latent_code], dim=1)
            specular_features = torch.cat([specular_features, latent_code], dim=1)
            blending_features = torch.cat([blending_features, latent_code], dim=1)
        if self.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            normal = torch.matmul(R_bwd, normal.unsqueeze(-1)).squeeze(-1)
            # view_noise_scale = self.cfg.get('view_noise', 0.)
            # if self.training and view_noise_scale > 0.:
            #     view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
            #                               dtype=torch.float32,
            #                               device=normal.device).transpose(0, 1)
            #     normal = torch.matmul(normal, view_noise)
        normal_embed = self.sh_embed(normal)
        blending_features = torch.cat([blending_features, normal_embed], dim=1)

        return diffuse_features, specular_features, blending_features

    def forward(self, gaussians, camera):
        shading_normal_features = self.compose_shading_input(gaussians, camera)
        # shading_normal_offset prediction
        # offset -> [-1, 1]
        shading_normal_offset = self.shading_normal_offset_activation(self.shading_normal_mlp(shading_normal_features))
        diffuse_features, specular_features, blending_features = self.compose_input(gaussians, camera, shading_normal_offset)
        diffuse_output = self.color_activation(self.diffuse_mlp(diffuse_features))
        specular_output = self.color_activation(self.specular_mlp(specular_features))
        blending_output = self.blending_activation(self.blending_mlp(blending_features))
        # linear composition
        if self.texture_mode == "diffuse":
            # only diffuse
            color_precomp = diffuse_output
        elif self.texture_mode == "specular":
            #only specular
            color_precomp = specular_output
        else:
            color_precomp = (1 - blending_output) * diffuse_output + blending_output * specular_output
        # change to sRGB space and map to [0, 1]
        shading_normal_offset_loss = torch.norm(shading_normal_offset, p=1, dim=1).mean()
        loss_reg ={
            "shading_normal_offset_loss": shading_normal_offset_loss
        }
        return color_precomp, loss_reg

def get_texture(cfg, metadata):
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,
        "fusion_mlp": FusionMLP
    }
    return model_dict[name](cfg, metadata)