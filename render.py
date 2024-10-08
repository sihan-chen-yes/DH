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
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random, transform_normals, srgb_to_linear
from scene import GaussianModel

from utils.general_utils import Evaluator, PSEvaluator
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
import open3d as o3d

import hydra
from omegaconf import OmegaConf
import wandb

from utils.general_utils import colormap


def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []

        examples = []

        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, config.opt, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

            # 2dgs
            rend_normal_image = render_pkg["rend_normal"]
            rend_normal_image = transform_normals(rend_normal_image, view.world_view_transform.T)

            rend_dist_image = render_pkg["rend_dist"]
            rend_dist_image = colormap(rend_dist_image.cpu().numpy()[0])

            surf_depth_image = render_pkg["surf_depth"]
            norm = surf_depth_image.max()
            surf_depth_image = surf_depth_image / norm
            surf_depth_image = colormap(surf_depth_image.cpu().numpy()[0], cmap='turbo')

            surf_normal_image = render_pkg["surf_normal"]
            surf_normal_image = transform_normals(surf_normal_image, view.world_view_transform.T)


            wandb_img = wandb.Image(opacity_image[None],
                                    caption=config['name'] + "_view_{}/render_opacity".format(view.image_name))
            examples.append(wandb_img)

            wandb_img = wandb.Image(rendering[None], caption='render_{}'.format(view.image_name))
            examples.append(wandb_img)

            # 2dgs
            wandb_img = wandb.Image(rend_normal_image[None],
                                    caption=config['name'] + "_view_{}/rend_normal_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(rend_dist_image[None],
                                    caption=config['name'] + "_view_{}/rend_dist_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(surf_depth_image[None],
                                    caption=config['name'] + "_view_{}/surf_depth_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(surf_normal_image[None],
                                    caption=config['name'] + "_view_{}/surf_normal_image".format(view.image_name))
            examples.append(wandb_img)

            wandb.log({config['name'] + "_images": examples})
            examples.clear()

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)



def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []

        examples = []

        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, config.opt, background,
                                compute_loss=False, return_opacity=False)

            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

            gt_mask = view.data['original_mask']

            # 2dgs
            rend_normal_image = render_pkg["rend_normal"]
            rend_normal_image = transform_normals(rend_normal_image, view.world_view_transform.T)
            # prune rendered normal
            rend_normal_image = rend_normal_image * gt_mask

            rend_dist_image = render_pkg["rend_dist"]
            rend_dist_image = colormap(rend_dist_image.cpu().numpy()[0])

            surf_depth_image = render_pkg["surf_depth"]
            norm = surf_depth_image.max()
            surf_depth_image = surf_depth_image / norm
            surf_depth_image = colormap(surf_depth_image.cpu().numpy()[0], cmap='turbo')

            surf_normal_image = render_pkg["surf_normal"]
            surf_normal_image = transform_normals(surf_normal_image, view.world_view_transform.T)
            # prune depth normal
            surf_normal_image = surf_normal_image * gt_mask

            gt = view.original_image[:3, :, :]

            wandb_img = wandb.Image(opacity_image[None],
                                    caption=config['name'] + "_view_{}/render_opacity".format(view.image_name))
            examples.append(wandb_img)

            wandb_img = wandb.Image(rendering[None], caption='render_{}'.format(view.image_name))
            examples.append(wandb_img)

            wandb_img = wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))
            examples.append(wandb_img)

            # 2dgs
            wandb_img = wandb.Image(rend_normal_image[None],
                                    caption=config['name'] + "_view_{}/rend_normal_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(rend_dist_image[None],
                                    caption=config['name'] + "_view_{}/rend_dist_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(surf_depth_image[None],
                                    caption=config['name'] + "_view_{}/surf_depth_image".format(view.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(surf_normal_image[None],
                                    caption=config['name'] + "_view_{}/surf_normal_image".format(view.image_name))
            examples.append(wandb_img)

            wandb.log({config['name'] + "_images": examples})
            examples.clear()

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)

def extract_mesh(config):
    model = config.model
    dataset_config = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    reconstruct_dir = os.path.join(config.exp_dir, config.suffix)

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    load_ckpt = config.get('load_ckpt', None)
    if load_ckpt is None:
        load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
    scene.load_checkpoint(load_ckpt)

    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # iter_start = torch.cuda.Event(enable_timing=True)
    # iter_end = torch.cuda.Event(enable_timing=True)
    #
    # evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

    gaussExtractor = GaussianExtractor(gaussians, scene, render, pipe, opt, bg_color=background)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    os.makedirs(reconstruct_dir, exist_ok=True)
    reconstruct_frames = scene.test_dataset.reconstruct_frames

    gaussExtractor.estimate_bounding_sphere(scene.test_dataset)
    depth_trunc = (gaussExtractor.radius * 2.0) if pipe.depth_trunc < 0 else pipe.depth_trunc
    voxel_size = (depth_trunc / pipe.mesh_res) if pipe.voxel_size < 0 else pipe.voxel_size
    sdf_trunc = 5.0 * voxel_size if pipe.sdf_trunc < 0 else pipe.sdf_trunc

    for frame in range(reconstruct_frames[0], reconstruct_frames[1], reconstruct_frames[2]):
        print(f"exporting mesh at frame {frame}")
        gaussExtractor.reconstruction(scene.test_dataset, 0, frame)
        # extract the mesh and save

        name = f"{frame}.ply"
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
                                                   reconstruct_frame=frame)

        # o3d.io.write_triangle_mesh(os.path.join(reconstruct_dir, name), mesh)
        # print("mesh saved at {}".format(os.path.join(reconstruct_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=pipe.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(reconstruct_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(reconstruct_dir, name.replace('.ply', '_post.ply'))))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    # config.dataset.root_dir = '/cluster/courses/digital_humans/datasets/team_8/ZJUMoCap'
    config.dataset.root_dir = '../ZJUMoCap'
    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    elif config.mode == 'reconstruct':
        config.suffix = config.mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project=config.project,
        entity='digital-human-s24',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    elif config.mode == 'reconstruct':
        extract_mesh(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()