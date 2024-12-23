import numpy as np
import torch

from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(gaussians,stack,camera:Camera, cam_trans_delta = torch.zeros(3),cam_rot_delta= torch.zeros(3), converged_threshold=1e-4):
    tau = torch.cat([cam_trans_delta, cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R_mirror
    T_w2c[0:3, 3] = camera.T_mirror

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]
    camera.R_mirror = new_R
    camera.T_mirror = new_T

    converged = tau.norm() < converged_threshold

    camera.world_view_transform_mirror = torch.tensor(getWorld2View2(new_R.numpy(), new_T.numpy())).transpose(0, 1).cuda()
    camera.full_proj_transform_mirror = (camera.world_view_transform_mirror.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
    camera.camera_center_mirror= camera.world_view_transform_mirror.inverse()[3, :3]
   
    
    
    
    cur_mirror_view = camera.world_view_transform_mirror
    cur_view = camera.world_view_transform
    # before =  get_mirror_transform(gaussians.mirror_equ_params[0],gaussians.mirror_equ_params[1],gaussians.mirror_equ_params[2],gaussians.mirror_equ_params[3])
    # print(before)
    w2c = cur_view.transpose(0, 1)
    new_mirror_transform = torch.matmul(cur_mirror_view.transpose(0, 1).inverse(),
                                        cur_view.inverse().transpose(0, 1).inverse())
    gaussians.checkpoint_mirror_transform = new_mirror_transform
    
    for cur in stack:
        cur.init_mirror_transform(gaussians.checkpoint_mirror_transform)
    return converged,camera


def update_pose_for_origin(camera:Camera, cam_trans_delta = torch.zeros(3),cam_rot_delta= torch.zeros(3), converged_threshold=1e-4):
    tau = torch.cat([cam_trans_delta, cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = torch.from_numpy( camera.R)
    T_w2c[0:3, 3] = torch.from_numpy(camera.T)

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]
    camera.R = new_R.numpy()
    camera.T = new_T.numpy()

    converged = tau.norm() < converged_threshold

    camera.world_view_transform = torch.tensor(getWorld2View2(new_R.numpy(), new_T.numpy())).transpose(0, 1).cuda()
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    # camera.update_RT(new_R, new_T)


    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    return converged,camera
