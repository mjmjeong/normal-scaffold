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

import numpy as np

import subprocess
#cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
#result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

#os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_uilts import *
import imageio 

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "our_video_{}".format(iteration), "renders")
    video_path = os.path.join(model_path, name)

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []

    view = views[0]
    print(name, len(views))
    #render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    #render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    #imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    # render_path_spiral
    # render_path_spherical
    render_images = []
    render_depths = [] 
    #render_poses = render_path_spiral(views[:10], focal=1, N=100)
    #render_poses = generate_spherical_sample_path(views[:1])
    render_poses = generate_ellipse_path(views,n_frames=200)
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, return_normal_depth=True)
        rendering = render_pkg["render"]
        depth = render_pkg["depth_expected"] * -1 
        depth = (depth - depth.min())/(depth.max()-depth.min())
        
        render_images.append(rendering)
        render_depths.append(depth)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    depth_array = torch.stack(render_depths, dim=0).permute(0, 2, 3, 1).repeat(1,1,1,3)
    depth_array = (depth_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)

    video_name = os.path.join(video_path, 'ours_rgb.mp4')
    gif_name = os.path.join(video_path, 'ours_rgb.gif')
    gif_depth_name = os.path.join(video_path, 'ours_depth.gif')

    imageio.mimwrite(video_name, render_array, fps=50)
    imageio.mimsave(gif_name, render_array, fps=50)
    imageio.mimsave(gif_depth_name, depth_array, fps=50)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.fewshot_num = -1
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
