import torch
import torch.nn as nn

from utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from utils.general_utils import build_rotation, linear_to_srgb
from models.network_utils import VanillaCondMLP, IntegratedDirectionalEncoding
from pytorch3d.transforms import matrix_to_quaternion


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
        loss_reg ={

        }
        return color, loss_reg

class FusionMLP(ColorPrecompute):
    '''
    Spatial MLPs:
    diffuse_mlp(xyz, gs_features, geometry_normal_features, latent_features),
    tint_mlp(xyz, gs_features, geometry_normal_features, latent_features)
    bottleneck_mlp(xyz, gs_features, geometry_normal_features, latent_features)
    roughness_mlp(xyz, gs_features, geometry_normal_features, latent_features)

    Directional MLPs:
    specular_mlp(bottleneck, ide(roughness, reflection_features), dot_product_features, latent_features)
    shading_normal_mlp(bottleneck, ide(roughness, reflection_features), dot_product_features, latent_features)
    '''
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        d_spatial_in = cfg.feature_dim

        d_directional_in = cfg.bottleneck_feature_dim

        d_ide_in = 1 #roughtness dim


        self.use_xyz = cfg.get('use_xyz', False)
        # self.use_cov = cfg.get('use_cov', False)
        # self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        # self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.latent_dim = cfg.get('latent_dim', 0)

        self.use_ref = cfg.get('use_ref', False)
        self.use_dir_normal_dot = cfg.get('use_dir_normal_dot', False)
        self.texture_mode = cfg.get('texture_mode', 'fusion')
        self.use_shading_normal_offset = cfg.get('use_shading_normal_offset', False)

        if self.use_xyz:
            d_spatial_in += 3

        # if self.use_xyz:
        #     d_in += 3
        # if self.use_cov:
        #     d_in += 6 # only upper triangle suffice
        # if self.use_normal:
        #     d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            # d_specular_in += (self.sh_degree + 1) ** 2 - 1
            # d_tint_in += (self.sh_degree + 1) ** 2 - 1
            # d_shading_normal_in += (self.sh_degree + 1) ** 2 - 1
            d_dir_in = (self.sh_degree + 1) ** 2 - 1
            d_spatial_in += d_dir_in
            d_ide_in += d_dir_in
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        # if self.non_rigid_dim > 0:
        #     d_in += self.non_rigid_dim

        if self.latent_dim > 0:
            # d_diffuse_in += self.latent_dim
            # d_specular_in += self.latent_dim
            # d_tint_in += self.latent_dim
            # d_shading_normal_in += self.latent_dim
            d_spatial_in += self.latent_dim
            d_directional_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        if self.use_dir_normal_dot:
            d_directional_in += 1

        d_color_out = 3
        d_tint_out = 3
        d_bottleneck_out = cfg.bottleneck_feature_dim
        d_roughness_out = 1
        d_shading_normal_out = 3

        # Spatial MLPs
        self.diffuse_mlp = VanillaCondMLP(d_spatial_in, 0, d_color_out, cfg.mlp)
        self.tint_mlp = VanillaCondMLP(d_spatial_in, 0, d_tint_out, cfg.mlp)
        self.bottleneck_mlp = VanillaCondMLP(d_spatial_in, 0, d_bottleneck_out, cfg.mlp)
        self.roughness_mlp = VanillaCondMLP(d_spatial_in, 0, d_roughness_out, cfg.mlp)

        # reflection direction encoding
        self.ide = IntegratedDirectionalEncoding(d_ide_in, cfg.ide)

        d_directional_in += self.ide.n_output_dims
        # Directional MLPs
        self.specular_mlp = VanillaCondMLP(d_directional_in, 0, d_color_out, cfg.mlp)
        self.shading_normal_mlp = VanillaCondMLP(d_directional_in, 0, d_shading_normal_out, cfg.mlp)

        self.color_activation = nn.Sigmoid()
        self.tint_activation = nn.Sigmoid()
        self.roughness_activation = nn.Softplus()
        self.roughness_bias = cfg.get('roughness_bias', 1.0)
        self.shading_normal_offset_activation = nn.Tanh()
        self.normal_activation = torch.nn.functional.normalize

    def compose_spatial_input(self, gaussians, camera):
        spatial_features = gaussians.get_features.squeeze(-1)

        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            spatial_features = torch.cat([spatial_features, xyz_norm], dim=1)

        normal = self.normal_activation(build_rotation(gaussians.get_rotation)[:, :, -1])
        # transform normals back into canonical space
        if self.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            normal = torch.matmul(R_bwd, normal.unsqueeze(-1)).squeeze(-1)
        normal_embed = self.sh_embed(normal)
        spatial_features = torch.cat([spatial_features, normal_embed], dim=1)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(spatial_features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(spatial_features.shape[0], -1)
            spatial_features = torch.cat([spatial_features, latent_code], dim=1)

        return spatial_features


    def compose_directional_input(self, gaussians, camera, bottleneck_features, roughness):
        directional_features = bottleneck_features

        normal = self.normal_activation(build_rotation(gaussians.get_rotation)[:, :, -1])

        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(directional_features.shape[0], 1))
            # normalize
            dir_pp = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)

            if self.use_dir_normal_dot:
                # dot product under pose space (or canonical space)
                dir_normal_dot = torch.bmm(dir_pp.unsqueeze(1), normal.unsqueeze(2)).squeeze(2)
                directional_features = torch.cat([directional_features, dir_normal_dot], dim=1)

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
            dir_embed = self.ide(dir_pp, roughness)
            directional_features = torch.cat([directional_features, dir_embed], dim=1)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(directional_features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(directional_features.shape[0], -1)
            directional_features = torch.cat([directional_features, latent_code], dim=1)

        return directional_features

    def forward(self, gaussians, camera):
        spatial_features = self.compose_spatial_input(gaussians, camera)

        diffuse_output = self.color_activation(self.diffuse_mlp(spatial_features))
        tint_output = self.tint_activation(self.tint_mlp(spatial_features))
        bottleneck_output = self.bottleneck_mlp(spatial_features)
        roughness_output = self.roughness_activation(self.roughness_mlp(spatial_features) + self.roughness_bias)

        directional_features = self.compose_directional_input(gaussians, camera, bottleneck_output, roughness_output)
        specular_output = self.color_activation(self.specular_mlp(directional_features))

        shading_normal_offset_loss = 0
        if self.use_shading_normal_offset:
            # shading_normal_offset prediction
            # offset -> [-1, 1]
            shading_normal_offset = self.shading_normal_offset_activation(self.shading_normal_mlp(directional_features))
            rots = build_rotation(gaussians.get_rotation)
            # last column as normal
            rots[:, :, -1] = self.normal_activation(rots[:, :, -1] + shading_normal_offset)
            gaussians._rotation = matrix_to_quaternion(rots)
            shading_normal_offset_loss = torch.norm(shading_normal_offset, p=1, dim=1).mean()

        # linear composition
        if self.texture_mode == "diffuse":
            # only diffuse
            color_precomp = diffuse_output
        elif self.texture_mode == "specular":
            #only specular
            color_precomp = specular_output
        else:
            color_precomp = diffuse_output + tint_output * specular_output


        # epsilon = 1e-6
        # opacity = torch.clamp(gaussians.get_opacity, epsilon, 1 - epsilon)
        # opacity_constraint = torch.mean(torch.log(opacity) + torch.log(1 - opacity))
        opacity_constraint = torch.mean(gaussians.get_opacity ** 2 + (1 - gaussians.get_opacity) ** 2)

        loss_reg ={
            "shading_normal_offset": shading_normal_offset_loss,
            "opacity_constraint": opacity_constraint
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