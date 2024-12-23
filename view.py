import sys
from scene import Scene 
from scene.mirror_gaussian_model import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer_mirror import render, network_gui
# from utils.image_utils import render_net_image
import torch
import subprocess
from torchvision.transforms import Resize
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from utils.image_utils import psnr
import time
from scene.cameras import Camera
import copy
from zzupdate_camera import update_pose
import matplotlib
matplotlib.use('TkAgg')
# 开启交互模式
# plt.ion()
fig, ax = plt.subplots()
camera_id = 10
rot_delta = torch.tensor([0,0,0]).float()
trans_delta = torch.tensor([0,0,0]).float()
viewpoints = None
Sensitivity = 0.01
MoveSensitivity = 0.05*4
def view(dataset, pipe, iteration,opt):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    (model_params, first_iter) = torch.load("/home/mousechannel/project/gs/zzlast/moMirrorGS/output/superlou/chkpnt10000.pth")
  
    gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    viewpoints = scene.getTrainCameras().copy()

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                # try:
                    net_image_bytes = None
                    # custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    custom_cam, do_training,   keep_alive, scaling_modifier,camera_id= network_gui.receive()
                    if custom_cam != None:
                        if camera_id >1:
                            cam =  viewpoints[camera_id]
                            custom_cam = cam
                            render_pkg = render(custom_cam, gaussians, pipe, background, 1.0)
                            image = render_pkg["render"]
                            mirror_render_pkg = render(custom_cam, gaussians, pipe, background, 1.0,mirror_transform=True)

                            mirror_image = mirror_render_pkg["render"]
                            gt_image = cam.original_image.cuda()  
                            gt_mirror_mask = cam.gt_alpha_mask.expand_as(gt_image) 
                            # gt_mirror_mask = resize_transform(gt_mirror_mask)
                            

                            image = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask

                            

                        else:
                            render_pkg = render(custom_cam, gaussians, pipe, background, 1.0)
                            image = render_pkg["render"]
                        net_image = render_net_image(image, dataset.render_items, 'None', custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                # except Exception as e:
                #     raise e
                #     print('Viewer closed')
                #     exit(0)






def myview(dataset, pipe, iteration,opt,checkpoint_path):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    (model_params, first_iter) = torch.load(checkpoint_path)
  
    gaussians.restore(model_params, opt)
    scene.init_camera_mirror(gaussians)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    global viewpoints
    viewpoints= scene.getTrainCameras().copy()

    
    

     
    cam =  viewpoints[camera_id]
    im = ax.imshow(viewpoints[10].original_image.cpu().permute(1,2,0).repeat(1,2,1) .numpy())

    def change_mirror_transform(stack, id,gaussians):
        if gaussians.checkpoint_mirror_transform is None:
            gaussians.checkpoint_mirror_transform = torch.tensor([[8.7663e-01, -1.5401e-01, 4.5585e-01, 1.1568e+01],
                                             [-2.5607e-01, 6.5276e-01, 7.1297e-01, 1.1133e+01],
                                             [4.0737e-01, 7.4174e-01, -5.3279e-01, -1.0598e+01],
                                             [-1.6619e-08, 2.9438e-09, -3.6980e-09, 1.0000e+00]]).cuda() 
            for camera in stack:
                camera.init_mirror_transform(gaussians.checkpoint_mirror_transform)
        # cur_camera = stack[id]
        # cur_mirror_view = cur_camera.world_view_transform_mirror
        # cur_view = cur_camera.world_view_transform
        # # before =  get_mirror_transform(gaussians.mirror_equ_params[0],gaussians.mirror_equ_params[1],gaussians.mirror_equ_params[2],gaussians.mirror_equ_params[3])
        # # print(before)
        # w2c = cur_view.transpose(0, 1)
        # new_mirror_transform = torch.matmul(cur_mirror_view.transpose(0, 1).inverse(),
        #                                     cur_view.inverse().transpose(0, 1).inverse())
        # # new_mirror_transform = torch.tensor([[8.7663e-01, -1.5401e-01, 4.5585e-01, 1.1568e+01],
        # #                                      [-2.5607e-01, 6.5276e-01, 7.1297e-01, 1.1133e+01],
        # #                                      [4.0737e-01, 7.4174e-01, -5.3279e-01, -1.0598e+01],
        # #                                      [-1.6619e-08, 2.9438e-09, -3.6980e-09, 1.0000e+00]]).cuda()
        # gaussians.checkpoint_mirror_transform = new_mirror_transform
        # new_mirror_transform = gaussians.checkpoint_mirror_transform

        # cur_camera.world_view_transform_mirror = torch.matmul(w2c, new_mirror_transform.inverse()).transpose(0, 1)
        # cur_camera.R_mirror = cur_camera.world_view_transform_mirror[:3, :3]
        # cur_camera.T_mirror = cur_camera.world_view_transform_mirror[3, :3]

        # cur_camera.full_proj_transform_mirror = (cur_camera.world_view_transform_mirror.unsqueeze(0).bmm(
        #     cur_camera.projection_matrix.unsqueeze(0))).squeeze(0)
        # cur_camera.camera_center_mirror = cur_camera.world_view_transform_mirror.inverse()[3, :3]


        # for camera in stack:
            # mirror_transform = get_mirror_transform(gaussians.mirror_equ_params[0], gaussians.mirror_equ_params[1],
            #                                         gaussians.mirror_equ_params[2], gaussians.mirror_equ_params[3])
            # if gaussians.checkpoint_mirror_transform is not None:
            #     mirror_transform = gaussians.checkpoint_mirror_transform

            # w2c = camera.world_view_transform.transpose(0, 1)
        #     if camera.world_view_transform_mirror is None:
        #         camera.world_view_transform_mirror = torch.matmul(w2c, mirror_transform.inverse()).transpose(
        #             0, 1)
        #         camera.R_mirror = camera.world_view_transform_mirror[:3, :3]
        #         camera.T_mirror = camera.world_view_transform_mirror[3, :3]
        #     if camera.full_proj_transform_mirror is None:
        #         camera.full_proj_transform_mirror = (
        #             camera.world_view_transform_mirror.unsqueeze(0).bmm(
        #                 camera.projection_matrix.unsqueeze(0))).squeeze(0)
        #     if camera.camera_center_mirror is None:
        #         camera.camera_center_mirror = camera.world_view_transform_mirror.inverse()[3, :3]
        #     # camera = stack[i]
        #     cur_mirror_view = camera.world_view_transform_mirror
        #     cur_view = camera.world_view_transform
        #     # before =  get_mirror_transform(gaussians.mirror_equ_params[0],gaussians.mirror_equ_params[1],gaussians.mirror_equ_params[2],gaussians.mirror_equ_params[3])
        #     # print(before)
        #     w2c = cur_view.transpose(0, 1)
        #     new_mirror_transform = torch.matmul(cur_mirror_view.transpose(0,1).inverse(),cur_view.inverse().transpose(0, 1).inverse())
        #     gaussians.force_mirror_transform = new_mirror_transform
        #     gaussians.force_mirror_transform = torch.tensor([[-9.0654e-01, -1.0663e-01,  4.0843e-01,  3.2349e+00],
        # [-1.0663e-01,  9.9404e-01,  2.2843e-02,  1.8092e-01],
        # [ 4.0843e-01,  2.2843e-02,  9.1249e-01, -6.9299e-01],
        # [ 5.0772e-07, -2.1718e-07, -4.2657e-07,  1.0000e+00]]).cuda()
        #     new_mirror_transform = gaussians.force_mirror_transform



            # camera.world_view_transform_mirror = torch.matmul(w2c, mirror_transform.inverse()).transpose(0, 1)
            # camera.R_mirror = camera.world_view_transform_mirror[:3,:3]
            # camera.T_mirror = camera.world_view_transform_mirror[3,:3]

            # camera.full_proj_transform_mirror = (camera.world_view_transform_mirror.unsqueeze(0).bmm(
            #     camera.projection_matrix.unsqueeze(0))).squeeze(0)
            # camera.camera_center_mirror = camera.world_view_transform_mirror.inverse()[3, :3]

        print("transform changed")
        print(gaussians.checkpoint_mirror_transform)

        # new_mirror_transform = torch.tensor([[8.7663e-01, -1.5401e-01, 4.5585e-01, 1.1568e+01],
        #                                  [-2.5607e-01, 6.5276e-01, 7.1297e-01, 1.1133e+01],
        #                                  [4.0737e-01, 7.4174e-01, -5.3279e-01, -1.0598e+01],
        #                                  [-1.6619e-08, 2.9438e-09, -3.6980e-09, 1.0000e+00]]).cuda()
        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_with_mirror_transform_" + str(iteration) + ".pth")
        # print(gaussians.force_mirror_transform)
    def just_show(cam,gaussians,pipe,background):
        render_pkg = render(cam, gaussians, pipe, background, 1.0)
        image = render_pkg["render"]
        mirror_render_pkg = render(cam, gaussians, pipe, background, 1.0,use_mirror_transform=True)

        mirror_image = mirror_render_pkg["render"]
        gt_image = cam.original_image.cuda()  
        gt_mirror_mask = cam.gt_alpha_mask.expand_as(gt_image) 
        # gt_mirror_mask = resize_transform(gt_mirror_mask)

        mirror_psnr =  psnr(mirror_image * gt_mirror_mask, gt_image * gt_mirror_mask)
        print(mirror_psnr.mean())
        

        image = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask
        image = image.cpu().permute(1,2,0).detach().numpy()
        image_all = np.concatenate([image, gt_image.cpu().permute(1,2,0).numpy()], axis=1)
        return np.clip(image_all, 0, 1)

        
        
 
    def on_key_press(event):
        global camera_id
        global viewpoints
        global  Sensitivity
        global MoveSensitivity
        # if event.key == 'control':
        #     print(f'你按下了 {event.key} 键')
        #     Sensitivity = 0.1
        #     MoveSensitivity = 0.1
        if event.key == 'left':
            print(f'你按下了 {event.key} 键')
            camera_id = camera_id+1
            # camera_id = min(camera_id,250)
            if camera_id >250:
                camera_id = 250
            img = just_show(viewpoints[camera_id],gaussians, pipe, background)
            im.set_data(img)
             
      
        if event.key == 'right':
            print(f'你按下了 {event.key} 键')
            camera_id = camera_id-1
            if camera_id <1:
                camera_id = 1
            # camera_id = max(0,camera_id)
            img = just_show(viewpoints[camera_id],gaussians, pipe, background)
            im.set_data(img)

        if event.key == ' ':
            change_mirror_transform(viewpoints,camera_id,gaussians)
            
            
        if event.key == 'j':
            _,new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id],cam_trans_delta =  torch.tensor([MoveSensitivity,0,0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera,gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'l':
            _,new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id],cam_trans_delta =  torch.tensor([-MoveSensitivity,0,0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera,gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'i':
            _,new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id],cam_trans_delta =  torch.tensor([0,MoveSensitivity,0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera,gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'k':
            _,new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id],cam_trans_delta =  torch.tensor([0,-MoveSensitivity,0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera,gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'u':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_trans_delta=torch.tensor([0, 0, -MoveSensitivity]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'o':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_trans_delta=torch.tensor([0, 0, MoveSensitivity]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)


        if event.key == 'home':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([0, 0, Sensitivity]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'end':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([0, 0, -Sensitivity]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)

        if event.key == 'delete':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([Sensitivity, 0, 0 ]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'pagedown':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([-Sensitivity, 0, 0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'insert':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([0, Sensitivity, 0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)
        if event.key == 'pageup':
            _, new_camera = update_pose(gaussians,viewpoints,viewpoints[camera_id], cam_rot_delta=torch.tensor([0, -Sensitivity, 0]).float())
            viewpoints[camera_id] = new_camera
            img = just_show(new_camera, gaussians, pipe, background)
            im.set_data(img)

        plt.title(viewpoints[camera_id].image_name)

        fig.canvas.draw_idle()
        # plt.pause(0.1)

    cids = fig.canvas.callbacks.callbacks.get('key_press_event', [])
    for cid in cids.copy():
        fig.canvas.mpl_disconnect(cid)
    cid = fig.canvas.mpl_connect('key_press_event', on_key_press)


    def on_button_press_event(event):
        global  Sensitivity
        global MoveSensitivity

        Sensitivity = 0.5
        MoveSensitivity = 0.4
    def on_button_release_event(event):
        global  Sensitivity
        global MoveSensitivity

        Sensitivity = 0.02
        MoveSensitivity = 0.1


    cid_release = fig.canvas.mpl_connect('button_press_event', on_button_press_event)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_button_release_event)

    plt.show()
    plt.pause(100)  



                         

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=10001)
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    network_gui.init(args.ip, args.port)
    
    # view(lp.extract(args), pp.extract(args), args.iteration,opt.extract(args))

    myview(lp.extract(args), pp.extract(args), args.iteration,opt.extract(args))

    print("\nViewing complete.")