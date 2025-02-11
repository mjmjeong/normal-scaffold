import torch
import os
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsplat.rendering import rasterization, rasterization_inria_wrapper
import torchvision 

save_dir = './observation/tmp'
save_dict = torch.load(os.path.join('./observation/asset', 'info.pt'))
xyz = save_dict['xyz']
color = save_dict['color']
opacity = save_dict['opacity']
scaling = save_dict['scaling']
rot = save_dict['rot']
height = save_dict['height']
width = save_dict['width'] 

tanfovx = save_dict['tanfovx'] 
tanfovy = save_dict['tanfovy'] 

viewmatrix = save_dict['viewmatrix'] 
full_projmatrix = save_dict['full_projmatrix']

bg = save_dict['bg']
campos = save_dict['campos']

intrinsic = save_dict['projection_matrix']
FoVx = save_dict['FoVx'] 
FoVy = save_dict['FoVy'] 

######################
# 3DGS: inria
#####################
raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_projmatrix,
        sh_degree=1,
        campos=campos,
        prefiltered=False,
        debug=False
    )
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
screenspace_points = torch.zeros_like(xyz, requires_grad=True, device="cuda") + 0

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
torchvision.utils.save_image(rendered_image, os.path.join(save_dir, 'result_3dgs_inria.png'))

##########################
# 3DGS: gsplat
##########################
Ks = torch.eye(3).cuda()
Ks[0, 0] = width / (2*tanfovx)
Ks[1, 1] = height /(2*tanfovy)
Ks[0,-1] = width/2
Ks[1,-1] = height/2

viewmatrix = viewmatrix.transpose(0,1)

render_colors, render_alphas, info = rasterization(
            means=xyz,
            quats=rot, # check quats
            scales=scaling,
            opacities=opacity[:,0],
            colors=color[None],
            sh_degree=None, 
            viewmats=viewmatrix[None],
            Ks=Ks[None],
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            render_mode='RGB', 
            distributed=False,
            backgrounds=bg[None],
            near_plane=0.01,
            far_plane=100   
        )

import torchvision
image = render_colors[:,:,:,:3]
import numpy as np
torchvision.utils.save_image(image.permute(0,3,1,2), os.path.join(save_dir, 'results_3dgs_gsplat.png'))