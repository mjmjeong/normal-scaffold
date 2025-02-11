import gsplat
import torch
from torch.nn import functional as F
import mediapy as media
import torchvision

import math
from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

num_gaussians = 1000
means = F.normalize(torch.randn(num_gaussians, 3, device="cuda"), dim=-1)
means[..., 2] = torch.rand(num_gaussians) * 3 + 2
scales = torch.rand_like(means) * 0.1 + 0.05
quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")[None, :].expand(
    num_gaussians, 4
)
opacities = torch.rand_like(scales[..., 0])
rgbs = torch.rand_like(means)

num_cameras = 4
viewmats = torch.eye(4, device="cuda")[None].expand(num_cameras, 4, 4)
Ks = torch.tensor(
    [
        [300.0, 0.0, 150.0],
        [0.0, 300.0, 100.0],
        [0.0, 0.0, 1.0],
    ],
    device="cuda",
)[None, :, :].expand(num_cameras, 3, 3)
width, height = 300, 200

"""
features, alphas, normals, surf_normals, dist_loss, depth_median, _ = gsplat.rasterization_2dgs(
    means=means,
    quats=quaternions,
    opacities=opacities,
    colors=rgbs,
    scales=scales,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    render_mode="RGB+D",
)

image = features[..., :3].permute(0,3,1,2)
depth = features[..., 3:].permute(0,3,1,2).repeat(1,3,1,1)
depth = depth / 5

depth_median =  depth_median.permute(0,3,1,2).repeat(1,3,1,1)/5
normals = normals.permute(0,3,1,2)*0.5 + 0.5
surf_normals = surf_normals.permute(0,3,1,2)*0.5+0.5

save_image = torch.cat((image, depth,  depth_median, normals, surf_normals), 2)
torchvision.utils.save_image(save_image, 'observation/tmp/results_2dgs.png')

scales[:,-1] = 0

features, rendered_alphas, info = gsplat.rasterization(
    means=means,
    quats=quaternions,
    opacities=opacities,
    colors=rgbs,
    scales=scales,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    render_mode="RGB+D",
)


image = features[..., :3].permute(0,3,1,2)
depth = features[..., 3:].permute(0,3,1,2).repeat(1,3,1,1)
depth = depth / 5

#depth_median =  depth_median.permute(0,3,1,2).repeat(1,3,1,1)/5
#normals = normals.permute(0,3,1,2)*0.5 + 0.5
#surf_normals = surf_normals.permute(0,3,1,2)*0.5+0.5

save_image = torch.cat((image, depth), 2)
torchvision.utils.save_image(save_image, 'observation/tmp/results_3dgs_gsplat.png')


features, rendered_alphas, info = gsplat.rasterization_inria_wrapper(
    means=means,
    quats=quaternions,
    opacities=opacities,
    colors=rgbs,
    scales=scales,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    render_mode="RGB+D",
)

image = features[..., :3].permute(0,3,1,2)
#depth = features[..., 3:].permute(0,3,1,2).repeat(1,3,1,1)
#depth = depth / 5

#depth_median =  depth_median.permute(0,3,1,2).repeat(1,3,1,1)/5
#normals = normals.permute(0,3,1,2)*0.5 + 0.5
#surf_normals = surf_normals.permute(0,3,1,2)*0.5+0.5

#save_image = torch.cat((image, depth), 2)
torchvision.utils.save_image(image, 'observation/tmp/results_3dgs_original.png')
"""