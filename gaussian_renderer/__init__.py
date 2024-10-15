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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.point_utils import depth_to_normal
from utils.general_utils import build_rotation

def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=True,
           return_depth=True,
           return_render_normal=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    render_alpha = None
    if return_opacity:
        render_alpha, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_alpha = render_alpha[:1]

    #expected depth map
    surf_depth = None
    if return_depth:
        R = torch.from_numpy(data.R).to(device=opacity.device).type_as(means3D)
        T = torch.from_numpy(data.T).to(device=opacity.device).type_as(means3D)
        # points in camera coordinate frame
        points_cam = means3D @ R + T[None, :]
        depths = points_cam[:, 2][:, None].expand(-1, 3)
        surf_depth, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=depths,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        surf_depth = surf_depth[:1]
        surf_depth = (surf_depth / (render_alpha + 1e-12))
        surf_depth = torch.nan_to_num(surf_depth, 0, 0)
        surf_normal = depth_to_normal(data, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

    render_normal = None
    if return_render_normal:
        # first column as normal for GaMeS
        normal = build_rotation(pc.get_rotation)[:, :, 0]
        render_normal, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=normal,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_normal = (render_normal.permute(1, 2, 0) @ (data.world_view_transform[:3, :3].T)).permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": render_alpha,

            'rend_alpha': render_alpha,
            "surf_depth": surf_depth,
            'surf_normal': surf_normal,
            'rend_normal': render_normal,
            }
