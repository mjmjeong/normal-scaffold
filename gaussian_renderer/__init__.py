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
from einops import repeat
import os

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation
from utils.graphics_utils import normal_from_depth_image 

import gsplat
from gsplat.rendering import rasterization
        
def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
#def get_render_normal(viewpoint_cam, depth):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, out_depth=False, return_normal=False, radius=0, args=None):
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    # intrinsic & extrinsic
    Ks = torch.eye(3).cuda() # TODO
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    Ks[0, 0] = viewpoint_camera.image_width / (2*tanfovx)
    Ks[1, 1] = viewpoint_camera.image_height / (2*tanfovy)
    Ks[0,-1] = viewpoint_camera.image_width/2
    Ks[1,-1] = viewpoint_camera.image_height/2
    viewmats = viewpoint_camera.world_view_transform.transpose(0,1)
    
    # infos: rgb, normal, uncertainty
    # 0) depth (alpha-blending)
    depth = (xyz-viewpoint_camera.camera_center).norm(dim=1, keepdim=True)
    # 1) normal
    rotations_mat = build_rotation(rot)
    scales = scaling
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    normal = rotations_mat[indices, :, min_scales]
    view_dir = xyz - viewpoint_camera.camera_center
    normal = (
        normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]
    ) # world coord
    
    
    # 2) uncertainty
    sorted_scale, _ = torch.sort(scaling, dim=-1)
    uncert = sorted_scale[..., :1]
    features = torch.cat((color, depth, normal, uncert), -1)

    render_feats, render_alphas, info = rasterization(
            means=xyz,
            quats=rot, 
            scales=scaling*scaling_modifier,
            opacities=opacity[:,0],
            colors=features[None],
            sh_degree=None, 
            viewmats=viewmats[None],
            Ks=Ks[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            render_mode='RGB+ED', 
            distributed=False,
            backgrounds=None,
            near_plane=0.01,
            far_plane=100   
        )

    if retain_grad:
        info['means2d'].retain_grad()
    # others
    image = render_feats[..., :3].permute(0,3,1,2)[0]
    normals = render_feats[..., 4:7].permute(0,3,1,2)[0]
    normals = torch.nn.functional.normalize(normals, dim=0)
    uncert =render_feats[..., 7:8].permute(0,3,1,2)[0]
    depth_median = render_feats[..., -1:].permute(0,3,1,2)[0]

    # depth & normal
    depth_alpha = render_feats[..., 3:4]
    if args.depth_correlation_with_alpha:
        depth_alpha = depth_alpha / render_alphas.clamp(min=1e-10)
    
    if args.depth_to_normal_func == 'gsplat':
        render_normals_from_depth = gsplat.utils.depth_to_normal(depth_alpha, torch.linalg.inv(viewmats[None]), Ks[None]).squeeze(0).permute(2,0,1)
        depth_alpha = depth_alpha.permute(0,3,1,2)[0]
    elif args.depth_to_normal_func == 'gsurf':
        depth_alpha = depth_alpha.permute(0,3,1,2)[0]
        render_normals_from_depth = render_normal(viewpoint_camera, depth_alpha[0]) 
        
    # normal for local cam (camera direction)
    R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
    # normal_local = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
    normals_local = (R_w2c @ normals.reshape(3, -1)).reshape(3, viewpoint_camera.image_height, viewpoint_camera.image_width)
    
    if is_training:
        return_dict = {"render": image,
                "depth_median": depth_median,
                "depth_expected": depth_alpha, 
                "normal_expected": normals, # world: expected
                "normal_depth": render_normals_from_depth, # world: from depth
                "normal_local": normals_local, # cam: expected
                "normal_uncert": uncert,
                "viewspace_points": info['means2d'],
                "visibility_filter" : info['radii'][0] > 0,
                "radii": info['radii'][0],
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "info": info, 
                }
    else:
        return_dict = {"render": image,
                "depth_median": depth_median,
                "depth_expected": depth_alpha, 
                "normal_expected": normals, # world
                "normal_depth": render_normals_from_depth, # world
                "normal_local": normals_local,
                "normal_uncert": uncert,
                "viewspace_points": info['means2d'],
                "visibility_filter" : info['radii'][0] > 0,
                "radii": info['radii'][0],
                }
    
    return return_dict

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, return_normal_depth=False, radius=0, args=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    out= rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
        
    rendered_image, radii = out[0], out[1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return_dict = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return_dict = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }

    if return_normal_depth:
        # Get the predicted normal of the Scaffold-GS, code get from Gaussian-pro
        rotations_mat = build_rotation(rot)
        scales = scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        view_dir = xyz - viewpoint_camera.camera_center
        normal = (
            normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]
        )

        
        out = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=normal,
            opacities=opacity,
            scales=scales,
            rotations=rot,
            cov3D_precomp=None,
        )
        render_normal_out = out[0]
        render_normal_out = torch.nn.functional.normalize(render_normal_out, dim=0)
        return_dict['normal_expected'] = render_normal_out
    
        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        #normal_local = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
        normals_local = (R_w2c @ render_normal_out.reshape(3, -1)).reshape(3, viewpoint_camera.image_height, viewpoint_camera.image_width)
        return_dict['normal_local'] = normals_local
        
        ##############################################################################
        # uncertainty + depth
        depth = (xyz-viewpoint_camera.camera_center).norm(dim=1, keepdim=True)

        sorted_scale, _ = torch.sort(scaling, dim=-1)
        uncert = sorted_scale[..., :1]
        features = torch.cat((depth, uncert, uncert), -1)
        out = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = features,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
        
        rendered_depth_hand = out[0][:1, :, :,]
        rendered_uncert = out[0][1:2, :, :,]
        
        return_dict['depth_expected'] = rendered_depth_hand
        return_dict['normal_depth'] = render_normal(viewpoint_camera, rendered_depth_hand[0]) 
        return_dict['normal_uncert'] = rendered_uncert
    return return_dict

def render2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, radius=0, out_depth=True, return_normal=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = PlaneGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=True,
        debug=pipe.debug
    )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
    
    rotations_mat = build_rotation(rot)
    scales = scaling
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    normal = rotations_mat[indices, :, min_scales]

    view_dir = xyz - viewpoint_camera.camera_center
    normal = (
        normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]
    )

    # TODO check
    depth_z = view_dir[:, -1]
    local_distance = (normal * view_dir).sum(-1).abs()
    input_all_map = torch.ones((xyz.shape[0], 7)).cuda().float()
    input_all_map[:, :3] = normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    # uncertainty
    sorted_scale, _ = torch.sort(scaling, dim=-1)
    input_all_map[:, 5] = sorted_scale[..., 0] # uncertainty # TODO
 #   # add depth
    input_all_map[:, 6] = (xyz-viewpoint_camera.camera_center).norm(dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        means2D_abs = screenspace_points_abs,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        all_map = input_all_map, 
        cov3D_precomp = None)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    rendered_uncert = out_all_map[5:6, ]
    rendered_depth = out_all_map[6:7, ]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return_dict = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "rendered_normal": rendered_normal,
                "plane_depth": plane_depth,
                "rendered_distance": rendered_distance,
#                "rendered_uncert": rendered_uncert,
 #               "rendered_depth": rendered_depth
                }
    else:
        return_dict = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "rendered_normal": rendered_normal,
                "plane_depth": plane_depth,
                "rendered_distance": rendered_distance,
  #              "rendered_uncert": rendered_uncert,
  #              "rendered_depth": rendered_depth
                }

    # additional estimation
    depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
    return_dict.update({"depth_normal": depth_normal})
    return return_dict

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


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

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
