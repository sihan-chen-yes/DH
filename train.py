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
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
from utils.general_utils import colormap, transform_normals, get_boundary_mask
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
import open3d as o3d
import math
import hydra
from omegaconf import OmegaConf
import wandb
import lpips


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value

def get_lambda(opt, name, iteration):
    lbd = opt.get(f"lambda_{name}", 0.)

    loss_from_dict = {
        "normal": opt.normal_loss_from,
        "dist": opt.dist_loss_from,
        "opacity_constraint": opt.opacity_constraint_loss_from,

    }

    loss_until_dict = {


    }

    # constrain lbd via from_iteration and until_iteration
    if name in loss_from_dict.keys() and name not in loss_until_dict.keys():
        # fixed lambda
        loss_from_iteration = loss_from_dict[name]
        lbd = 0.0 if iteration < loss_from_iteration else lbd
    elif name in loss_until_dict.keys() and name not in loss_from_dict.keys():
        # fixed lambda
        loss_until_iteration = loss_until_dict[name]
        lbd = 0.0 if iteration >= loss_until_iteration else lbd
    elif name in loss_from_dict.keys() and name in loss_until_dict.keys():
        loss_from_iteration = loss_from_dict[name]
        loss_until_iteration = loss_until_dict[name]
        method = opt.opt.get("decay_method", "linear")
        # lambda decay as linear
        if method == "linear":
            lbd = lbd * ((loss_until_iteration - iteration) / (loss_until_iteration - loss_from_iteration))
        # lambda decay as exponential
        elif method == "exponential":
            lbd = lbd * math.exp(-(iteration - loss_from_iteration))
        # lambda decay as cosine
        elif method == "cosine":
            lbd = lbd * 0.5 * (1 + math.cos(math.pi * (iteration - loss_from_iteration) / (loss_until_iteration - loss_from_iteration)))

    return lbd


def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # scene.save_mesh_TBN_map()
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data = scene.train_dataset[data_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        gt_image = data.original_image.cuda()
        gt_mask = data.original_mask.cuda()

        if dataset.random_background:
            # sRGB
            background = torch.tensor(np.random.rand(3), dtype=torch.float32, device="cuda")
            # use random background
            gt_image = gt_image * gt_mask + background[:, None, None] * (1 - gt_mask)

        render_pkg = render(data, iteration, scene, pipe, opt, background, compute_loss=True, return_opacity=use_mask)
        # rendered img
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        dir_normal_constraint = render_pkg['dir_normal_constraint']

        if dataset.foreground_crop:
            boundary_mask = torch.from_numpy(get_boundary_mask(gt_mask)).cuda()
            boundary_mask_img = 1. - boundary_mask.to(torch.float32)
            # linear fusion only
            # care foreground and background, not care edge
            image = image * boundary_mask_img[None, :, :] + (1. - boundary_mask_img[None, :, :]) * background[:, None, None]
            gt_image = gt_image * boundary_mask_img[None, :, :] + (1. - boundary_mask_img[None, :, :]) * background[:, None, None]
            opacity = opacity * boundary_mask_img[None, :, :]
            gt_mask = gt_mask * boundary_mask_img[None, :, :]

            # if only care foreground loss
            rend_dist = rend_dist * gt_mask
            rend_normal = rend_normal * gt_mask
            surf_normal = surf_normal * gt_mask
            dir_normal_constraint = dir_normal_constraint * gt_mask

        lambda_l1 = C(iteration, config.opt.lambda_l1)
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        if lambda_perceptual > 0:
            # crop the foreground
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]

            loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1.-1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # 2dgs regularization
        lambda_normal = get_lambda(opt, "normal", iteration)
        lambda_dist = get_lambda(opt, "dist", iteration)
        #TODO
        lambda_dir_normal_constraint = get_lambda(opt, "dir_normal_constraint", iteration)

        # mask, only care foreground
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

        loss_normal = normal_error.mean()
        loss_dist = rend_dist.mean()

        loss_dir_normal_constraint = dir_normal_constraint.mean()


        loss += lambda_normal * loss_normal
        loss += lambda_dist * loss_dist
        loss += lambda_dir_normal_constraint * loss_dir_normal_constraint

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = get_lambda(opt, name, iteration)
            lbd = C(iteration, lbd)
            loss += lbd * value
        loss.backward()


        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/normal_loss': loss_normal.item(),
                'loss/dist_loss': loss_dist.item(),
                'loss/dir_normal_constraint_loss': loss_dir_normal_constraint.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator, pipe, opt, background)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)

def validation(iteration, testing_iterations, testing_interval, scene : Scene, evaluator, pipe, opt, background):
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
                          {'name': 'train', 'cameras' : [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            examples = []
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, pipe, opt, background, compute_loss=False, return_opacity=True)
                image = render_pkg["render"]

                gt_mask = data.original_mask.to("cuda")

                gt_image = data.original_image.to("cuda")
                # use random background
                gt_image = gt_image * gt_mask + background[:, None, None] * (1 - gt_mask)

                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

                #2dgs
                rend_normal_image = render_pkg["rend_normal"]
                rend_normal_image = transform_normals(rend_normal_image, data.world_view_transform.T)
                rend_dist_image = render_pkg["rend_dist"]
                rend_dist_image = colormap(rend_dist_image.cpu().numpy()[0])

                surf_depth_image = render_pkg["surf_depth"]
                norm = surf_depth_image.max()
                surf_depth_image = surf_depth_image / norm
                surf_depth_image = colormap(surf_depth_image.cpu().numpy()[0], cmap='turbo')

                surf_normal_image = render_pkg["surf_normal"]
                surf_normal_image = transform_normals(surf_normal_image, data.world_view_transform.T)

                wandb_img = wandb.Image(opacity_image[None],
                                        caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/render".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                    data.image_name))
                examples.append(wandb_img)

                #2dgs
                wandb_img = wandb.Image(rend_normal_image[None], caption=config['name'] + "_view_{}/rend_normal_image".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(rend_dist_image[None], caption=config['name'] + "_view_{}/rend_dist_image".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(surf_depth_image[None], caption=config['name'] + "_view_{}/surf_depth_image".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(surf_normal_image[None], caption=config['name'] + "_view_{}/surf_normal_image".format(data.image_name))
                examples.append(wandb_img)

                l1_test += l1_loss(image, gt_image).mean().double()
                metrics_test = evaluator(image, gt_image)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                wandb.log({config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)


    config.dataset.root_dir = '../ZJUMoCap'
    # set wandb logger
    wandb_name = config.name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        entity='digital-human-s24',
        project=config.project,
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
