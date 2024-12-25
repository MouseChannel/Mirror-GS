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

import copy
import os
from random import randint
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.mirror_gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from torchvision.utils import save_image
import trimesh

import open3d as o3d
import numpy as np

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False,use_mirror_transform=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    
    viewmatrix = viewpoint_camera.world_view_transform
    projmatrix = viewpoint_camera.full_proj_transform
    campos = viewpoint_camera.camera_center
    if use_mirror_transform  :
        # mirror_transform = pc.mirror_transform_model.get_cur_mirror_transform()
        # if pc.checkpoint_mirror_transform is None:
        #     mirror_transform = pc.get_mirror_transform
        # else:
        #     mirror_transform = pc.checkpoint_mirror_transform
        mirror_transform =   pc.get_mirror_transform          
        w2c = viewmatrix.transpose(0, 1) # Q_o
        viewmatrix = torch.matmul(w2c, mirror_transform.inverse()).transpose(0, 1)
        projmatrix = (viewmatrix.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
        campos = viewmatrix.inverse()[3, :3]
        
    
    
    

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix, 
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=campos,

        prefiltered=False,
        # debug=pipe.debug,
        debug=False,

        # antialiasing=pipe.antialiasing
        antialiasing=False

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
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
        
        
    if use_mirror_transform and pc.checkpoint_mirror_transform is not None:
    #     # opacity = opacity * (1 - pc.get_mirror_opacity)
        mask = get_distance_to_plane_mask(pc)
        means3D = means3D[mask]
        means2D = means2D[mask]
        shs = shs[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        if cov3D_precomp:
            cov3D_precomp = cov3D_precomp[mask]


    try:
        means3D.retain_grad()
    except:
        pass
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp and pc.checkpoint_mirror_transform is not None:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def remove_fly_points(points:torch.Tensor,vis= False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
    inlier_cloud = pcd.select_by_index(ind)
    if vis:
        o3d.visualization.draw_geometries([inlier_cloud])

    return torch.asarray(np.asarray( inlier_cloud.points))


def calculate_mirror_transform(viewpoint_stack,pc:GaussianModel,pipe,bg_color : torch.Tensor, model_path,vis =False ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    mirror_points_mask,scene_points_mask = get_mirrot_points(viewpoint_stack,bg_color,pc,model_path,vis=vis)

    mirror_points = remove_fly_points(pc.get_xyz[mirror_points_mask],vis=vis)
    # mirror_transform = pc.calculate_plane(mirror_points.numpy())
    # mirror_transform = calculate_plane(mirror_points.numpy() )
    from utils import ransac

    mirror_equ, mirror_pts_ids = ransac.Plane(mirror_points.numpy(), 0.05)

    # apply mask
    pc.mirror_equ_params = mirror_equ
    pc.scene_point_mask = scene_points_mask
    pc.mirror_points_mask = mirror_points_mask
    # pc._xyz = pc._xyz[pc.scene_point_mask]
    # pc._features_dc =  pc._features_dc[pc.scene_point_mask]
    # pc._features_rest =  pc._features_rest[pc.scene_point_mask]
    # pc._scaling =  pc._scaling[pc.scene_point_mask]
    # pc._rotation =  pc._rotation[pc.scene_point_mask]
    # pc._opacity =  pc._opacity[pc.scene_point_mask]
    # pc.max_radii2D = pc.max_radii2D[pc.scene_point_mask]


    # pc.means3D = pc.means3D[pc.scene_point_mask]
    # pc.means2D = pc.means2D[pc.scene_point_mask]
    # pc.shs = pc.shs[pc.scene_point_mask]
    # pc.opacity = pc.opacity[pc.scene_point_mask]
    # pc.scales = pc.scales[pc.scene_point_mask]
    # pc.rotations = pc.rotations[pc.scene_point_mask]
    # if cov3D_precomp:
    #     cov3D_precomp = cov3D_precomp[cov3D_precomp]

 

def get_mirrot_points(viewpoint_stack_,bg_color,pc,model_path,vis=False):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    shs = pc.get_features
    scales = pc.get_scaling
    rotations = pc.get_rotation
    # colors = pc.get_mirror_opacity.repeat(1, 3).clone()
    # colors = colors.detach().requires_grad_(True)
    # colors = pc.get_mirror_opacity.repeat(1, 3)
    # colors = colors.detach().requires_grad_(True)
    colors= torch.zeros_like(pc.get_xyz,device=pc.get_xyz.device,requires_grad=True)


    # colors.
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device, requires_grad=False)
    # viewpoint_stack =  copy.deepcopy(viewpoint_stack_)
    # viewpoint_stack2 = copy.deepcopy(viewpoint_stack_)

    for i in range(len(viewpoint_stack_)):
        viewpoint_camera = viewpoint_stack_[randint(0, len(viewpoint_stack_)-1)]
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            # projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            antialiasing=False,
            debug=True
            # pipe.debug

        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        grad, _,_= rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )



        mask = viewpoint_camera.gt_alpha_mask>0.5
        mask = mask.repeat(3,1,1)
        target = grad *  mask .cuda().float()

        loss = 1 * target.mean()


        loss.backward(retain_graph=True)


        mins = torch.min(colors.grad, dim=-1).values
        maxes = torch.max(colors.grad, dim=-1).values
        assert torch.allclose(mins , maxes), "Something is wrong with gradient calculation"
        gaussian_grads += (colors.grad).norm(dim=[1])
        colors.grad.zero_()

        mask_inverted = ~mask
        target = grad * mask_inverted.cuda()
        loss = 1 * target.mean()
        loss.backward(retain_graph=True)
        gaussian_grads -= (colors.grad).norm(dim=[1])
        colors.grad.zero_()





    mask_3d = gaussian_grads > 0
    if not vis:
        return mask_3d ,~mask_3d

    #vis
    means3Dn = pc.get_xyz[~mask_3d]
    means2Dn = screenspace_points[~mask_3d]
    opacityn = pc.get_opacity[~mask_3d]
    shsn = pc.get_features[~mask_3d]
    scalesn = pc.get_scaling[~mask_3d]
    rotationsn = pc.get_rotation[~mask_3d]
    for i in range(len(viewpoint_stack_)):
        viewpoint_camera = viewpoint_stack_[randint(0, len(viewpoint_stack_)-1)]
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            # projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            antialiasing=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        img, _ ,_= rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,

        )
        os.makedirs(f"{model_path}/outfor_remove_point", exist_ok=True)
        save_image(img, f"{model_path}/outfor_remove_point/{str(i)}.png")

        # for nomask
        img, _  ,_= rasterizer(
            means3D=means3Dn,
            means2D=means2Dn,
            shs=shsn,
            colors_precomp=None,
            opacities=opacityn,
            scales=scalesn,
            rotations=rotationsn,
            cov3D_precomp=None,
        )
        # save_image(img, "outtemp/" + str(i) + "no.png")
        save_image(img, f"{model_path}/outfor_remove_point/{str(i)}no.png")

    return mask_3d ,~mask_3d


def get_distance_to_plane_mask(pc:GaussianModel):
    a ,b,c = pc.mirror_equ_params[0],pc.mirror_equ_params[1],pc.mirror_equ_params[2]
    plane_nrm = torch.tensor([[a],[b],[c]]).cuda().float()
    mirror_point_mask = torch.mm(pc.get_xyz,plane_nrm)<0
    return mirror_point_mask.squeeze(-1)
     
    


 

