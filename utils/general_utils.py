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
import os
import sys
from datetime import datetime
import numpy as np
import random

import torch.nn as nn
import lpips
import cv2
from skimage.metrics import structural_similarity as compute_ssim

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond() -> torch.Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    if r.shape[-1] == 4:
        # quaternion to matrix
        R = build_rotation(r)
    else:
        R = r

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def fix_random(seed):
    if seed >= 0:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

# evaluation metrics
class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()

    def forward(self, inputs, targets):
        psnr = self.psnr(inputs, targets)
        ssim = self.ssim(inputs, targets)
        lpips_ = self.lpips(inputs, targets)
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_,
        }

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets) ** 2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        device = inputs.device
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            x, y, w, h = cv2.boundingRect(valid_mask.astype(np.uint8))
            img_pred = inputs[y:y + h, x:x + w]
            img_gt = targets[y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        # compute ssim
        ssim = compute_ssim(img_pred, img_gt, channel_axis=0)
        ssim = torch.tensor(ssim, device=device)
        return ssim


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        self.loss_fn_vgg.eval()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        if valid_mask is not None:
            x, y, w, h = cv2.boundingRect(valid_mask.cpu().numpy().astype(np.uint8))
            img_pred = inputs[:, y:y + h, x:x + w]
            img_gt = targets[:, y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        score = self.loss_fn_vgg(img_pred, img_gt, normalize=True)
        return score.flatten()

class PSEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.cuda()
        self.eval()

    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.unsqueeze(0)
        rgb_gt = rgb_gt.unsqueeze(0)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img

def transform_normals(normals, w2c):
    """ Convert world-space normal map into OpenGL camera space
    """
    # Convert from world to view frame
    normals = (normals.permute(1, 2, 0) @ w2c.T[:3,:3]).permute(2, 0, 1)
    # normalize norm length to 1
    normals = torch.nn.functional.normalize(normals, dim=0)
    # Convert OpenCV to OpenGL convention
    # y and z axis need flipped
    normals = normals * torch.tensor([1.0, -1.0, -1.0], device=normals.device)[:, None, None]
    # map to [0, 1] from [-1, 1]
    normals = (normals + 1) / 2

    return normals

def get_boundary_mask(mask, kernel_size = 5):
    """
    :param mask: np.uint8
    :param kernel_size:
    :return:
    """
    # mask_bk = mask.copy()
    # thres = 128
    # mask[mask < thres] = 0
    # mask[mask > thres] = 1
    mask = mask.squeeze(0)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_erode = cv2.erode(mask.cpu().numpy(), kernel)
    mask_dilate = cv2.dilate(mask.cpu().numpy(), kernel)
    boundary_mask = (mask_dilate - mask_erode) == 1
    # boundary_mask = np.logical_or(boundary_mask,
    #                               np.logical_and(mask_bk > 5, mask_bk < 250))

    # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
    # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
    # cv.waitKey(0)

    return boundary_mask

def linear_to_srgb(linear_color, eps=None):
    '''
    Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB.
    Parameters
    ----------
    linear_color
    eps

    Returns
    -------
    sRGB
    '''
    if eps is None:
        eps = torch.finfo(linear_color.dtype).eps
    srgb0 = 323 / 25 * linear_color
    srgb1 = (211 * torch.clamp(linear_color, min=eps)**(5 / 12) - 11) / 200
    return torch.where(linear_color <= 0.0031308, srgb0, srgb1)

def srgb_to_linear(srgb, eps=None):
    '''
    Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB.
    Parameters
    ----------
    srgb
    eps

    Returns
    -------
    linear_color
    '''
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)