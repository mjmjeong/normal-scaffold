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

#import sys
#from pathlib import Path
#sys.path.append(str(Path(__file__).absolute().parent))
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')



from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel, render_gsplat
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn.functional as F

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        
        # stable normal

        # 3DGS: check
        tmp = time.time()
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, return_normal=True, out_depth=True, args=args)
        time_3dgs = time.time() - tmp
        normal_expected = (render_pkg['normal_expected']*-1) * 0.5 + 0.5
        normal_depth = (render_pkg['normal_depth']*-1)* 0.5 + 0.5
        normal_local = (render_pkg['normal_local']*-1)* 0.5 + 0.5

        depth_expected = render_pkg['depth_expected']
        depth_expected = (depth_expected.max() - depth_expected) / depth_expected.max()
        image = render_pkg['render']
        uncert = render_pkg['normal_uncert']

        type_ = '3dgs'
        torchvision.utils.save_image(image, f'./tmp/viz_image_{type_}.png')
        torchvision.utils.save_image(depth_expected, f'./tmp/viz_depth_{type_}.png')
        torchvision.utils.save_image(normal_depth, f'./tmp/viz_normal_depth_{type_}.png')
        torchvision.utils.save_image(normal_expected, f'./tmp/viz_normal_{type_}.png')
        torchvision.utils.save_image(normal_local, f'./tmp/viz_normal_local_{type_}.png')
        torchvision.utils.save_image(uncert*50, f'./tmp/viz_normal_uncert_{type_}.png')

        # gsplat: check
        tmp = time.time()
        render_pkg_gspat = render_gsplat(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, args=args)
        time_gsplat = time.time() - tmp

        type_ = 'gsplat'
        normal_expected_gsplat = (render_pkg_gspat['normal_expected']*-1) * 0.5 + 0.5
        normal_depth_gsplat = (render_pkg_gspat['normal_depth']*-1)* 0.5 + 0.5
        normal_local_gsplat = (render_pkg_gspat['normal_local']*-1)* 0.5 + 0.5

        depth_expected_gsplat = render_pkg_gspat['depth_expected']
        depth_expected_gsplat = (depth_expected_gsplat.max() - depth_expected_gsplat) / depth_expected_gsplat.max()
        image_gsplat = render_pkg_gspat['render']

        uncert = render_pkg_gspat['normal_uncert']

        torchvision.utils.save_image(image_gsplat, f'./tmp/viz_image_{type_}.png')
        torchvision.utils.save_image(depth_expected_gsplat, f'./tmp/viz_depth_{type_}.png')
        torchvision.utils.save_image(normal_depth_gsplat, f'./tmp/viz_normal_depth_{type_}.png')
        torchvision.utils.save_image(normal_expected_gsplat, f'./tmp/viz_normal_{type_}.png')
        torchvision.utils.save_image(normal_local_gsplat, f'./tmp/viz_normal_local_{type_}.png')

        torchvision.utils.save_image(uncert*50, f'./tmp/viz_normal_uncert_{type_}.png')
        print('time', time_3dgs, time_gsplat)
        print('rgb:', F.l1_loss(image, image_gsplat).item())
        print('depth:', F.l1_loss(depth_expected, depth_expected_gsplat).item())
        print('normal_depth:', F.l1_loss(normal_depth, normal_depth_gsplat).item())
        print('normal:', F.l1_loss(normal_expected, normal_expected_gsplat).item())

        breakpoint()
        
        torch.cuda.synchronize(); t1 = time.time()

        t_list.append(t1-t0)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args : None):
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
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args=args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args=args)

if __name__ == "__main__":
    """
    example: 
    python observation/check_render.py --model_path outputs/dtu_search/baseline --skip_train
    """
    
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
    args.depth_to_normal_func = 'gsurf'
    args.depth_correlation_with_alpha = False

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
