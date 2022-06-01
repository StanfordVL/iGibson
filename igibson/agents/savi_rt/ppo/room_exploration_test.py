import torch
import cv2
import numpy as np
from PIL import Image
import igibson
import os
import copy
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config
from igibson.agents.savi_rt.models.rt_predictor import RTPredictor, NonZeroWeightedCrossEntropy
from igibson.agents.savi_rt.utils.utils import batch_obs
from igibson.agents.savi_rt.utils.parallel_env import ParallelNavEnv
from igibson.agents.savi_rt.utils.environment import AVNavRLEnv
from igibson.agents.savi_rt.utils.utils import to_tensor
from igibson.agents.savi_rt.utils import dataset
torch.autograd.set_detect_anomaly(True)


def save_checkpoint(ckpt_path, checkpoint=None):
    torch.save(
        checkpoint, os.path.join("/viscam/u/wangzz/avGibson/igibson/agents/savi_rt/models", ckpt_path)
    )
    
def load_checkpoint(checkpoint_path: str, *args, **kwargs) -> Dict:   
    return torch.load(checkpoint_path, *args, **kwargs)

        
def train():
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config_filename = "/viscam/u/wangzz/avGibson/igibson/agents/savi_rt/config/savi_rt_step1.yaml"
    config = parse_config(config_filename)
    maps_path = "/viscam/u/wangzz/iGibson/gibson2/data/ig_dataset/scenes/Rs_int/layout/"

    scene_ids = ["Rs_int"]
    Rs_size = 1000
    def load_env(scene_id):
        return AVNavRLEnv(config_file=config_filename, mode='headless', scene_id=scene_id)

    envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                     for sid in scene_ids], blocking=False)

    predictor = RTPredictor(config, device, envs.observation_space).to(device=device)
    optimizer = torch.optim.SGD(predictor.parameters(),
                                    lr=0.4,
                                    momentum=0.9,
                                    weight_decay=0.0001)
    rt_loss_fn = NonZeroWeightedCrossEntropy().to(device=device)
    
    predictor.train()
    
    output_size = predictor.rt_map_output_size
    floor = 0
    gt_rt = np.flip(np.array(Image.open(os.path.join(maps_path, "floor_semseg_{}.png".format(floor)))), axis=0)
    gt_trav = np.flip(np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor)))), axis=0)
    gt_rt_resized = cv2.resize(gt_rt, (output_size, output_size))
    padded_gt_rt = to_tensor(gt_rt_resized).view(-1).to(device)

    step = 400
    epochs = 10
    best_acc = None
    

    for epoch in range(epochs):
        observations = envs.reset()
        batch = batch_obs(observations)
        initial_pos = batch["initial_pose"][:, :3]
        initial_rpy = batch["initial_pose"][:, 3:6]
    #     pos_eframe = rotate_vector_3d(initial_pos - initial_pos, 0, 0, initial_rpy[2])
    #     orn_eframe = initial_rpy[2] - initial_rpy[2]

        for i in range(step):  # 150 steps, 15s world time
            action = envs.action_space.sample()
            actions = [action]
            outputs = envs.step(actions, train=True) # autoreset: false
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device=device)

            curr_pos = batch["pose_sensor"][:, :3].cpu().detach().numpy()
            curr_rpy = batch["pose_sensor"][:, 3:6].cpu().detach().numpy()     
            pos_on_the_map = np.flip((curr_pos[0, :2]/0.1*output_size/Rs_size + output_size / 2.0)).astype(np.int)
            pos_on_the_map = np.array([output_size-pos_on_the_map[1], pos_on_the_map[0]])
            pos_on_full_map = np.flip((curr_pos[0, :2]/0.1+Rs_size / 2.0)).astype(np.int)
            pos_on_full_map = np.array([Rs_size-pos_on_full_map[1], pos_on_full_map[0]])

            global_map = predictor.update(batch, dones) # save a list of local maps after feature alignment

            rt_loss = rt_loss_fn(global_map.view(-1, predictor.rooms), padded_gt_rt)
            optimizer.zero_grad()
            rt_loss.backward()
            optimizer.step() 
            if best_acc is None or rt_loss < best_acc:
                best_model_wts = copy.deepcopy(predictor.state_dict())
                save_checkpoint(f"ckpt.{epoch}.{i}.pth", best_model_wts)

    envs.close()
    
    
def val():   
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config_filename = "/viscam/u/wangzz/avGibson/igibson/agents/savi_rt/config/savi_rt_step1.yaml"
    config = parse_config(config_filename)
    maps_path = "/viscam/u/wangzz/iGibson/gibson2/data/ig_dataset/scenes/Rs_int/layout/"

    scene_ids = ["Rs_int"]
    Rs_size = 1000
    def load_env(scene_id):
        return AVNavRLEnv(config_file=config_filename, mode='headless', scene_id=scene_id)

    envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                     for sid in scene_ids], blocking=False)

    predictor = RTPredictor(config, device, envs.observation_space).to(device=device)
    optimizer = torch.optim.SGD(predictor.parameters(),
                                    lr=0.4,
                                    momentum=0.9,
                                    weight_decay=0.0001)
    rt_loss_fn = NonZeroWeightedCrossEntropy().to(device=device)
    
    checkpoint_path = "/viscam/u/wangzz/avGibson/igibson/agents/savi_rt/models/ckpt.10.180.pth"
    ckpt_dict = load_checkpoint(checkpoint_path, map_location="cpu")
    predictor.load_state_dict(ckpt_dict["state_dict"])
    predictor.eval()
    
    output_size = predictor.rt_map_output_size
    floor = 0
    gt_rt = np.flip(np.array(Image.open(os.path.join(maps_path, "floor_semseg_{}.png".format(floor)))), axis=0)
    gt_trav = np.flip(np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor)))), axis=0)
    gt_rt_resized = cv2.resize(gt_rt, (output_size, output_size))
    padded_gt_rt = to_tensor(gt_rt_resized).view(-1).to(device)
    
    global_map_view = torch.argmax(global_map, dim=2)
    global_map_view = global_map_view.view(output_size, output_size)
    global_map_view = global_map_view.cpu().detach().numpy()# /23*255
    global_map_view = global_map_view*1.0 / predictor.rooms * 255
    heatmap_pred = cv2.applyColorMap(global_map_view.astype(np.uint8), cv2.COLORMAP_HSV)
    heatmap_pred = Image.fromarray(heatmap_pred.astype(np.uint8))

    curr_gt = gt_trav.copy()
    curr_gt[pos_on_full_map[0]-5:pos_on_full_map[0]+5, pos_on_full_map[1]-5:pos_on_full_map[1]+5] = 128
    heatmap_gt = cv2.applyColorMap(curr_gt.astype(np.uint8), cv2.COLORMAP_PINK)
    heatmap_gt = Image.fromarray(heatmap_gt.astype(np.uint8))
    
    step = 400
    epochs = 1
    best_acc = None
    

    for epoch in range(epochs):
        observations = envs.reset()
        batch = batch_obs(observations)
        initial_pos = batch["initial_pose"][:, :3]
        initial_rpy = batch["initial_pose"][:, 3:6]
    #     pos_eframe = rotate_vector_3d(initial_pos - initial_pos, 0, 0, initial_rpy[2])
    #     orn_eframe = initial_rpy[2] - initial_rpy[2]

        for i in range(step):  # 150 steps, 15s world time
            action = envs.action_space.sample()
            actions = [action]
            outputs = envs.step(actions, train=True) # autoreset: false
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device=device)

            curr_pos = batch["pose_sensor"][:, :3].cpu().detach().numpy()
            curr_rpy = batch["pose_sensor"][:, 3:6].cpu().detach().numpy()     
            pos_on_the_map = np.flip((curr_pos[0, :2]/0.1*output_size/Rs_size + output_size / 2.0)).astype(np.int)
            pos_on_the_map = np.array([output_size-pos_on_the_map[1], pos_on_the_map[0]])
            pos_on_full_map = np.flip((curr_pos[0, :2]/0.1+Rs_size / 2.0)).astype(np.int)
            pos_on_full_map = np.array([Rs_size-pos_on_full_map[1], pos_on_full_map[0]])

            global_map = predictor.update(batch, dones) # save a list of local maps after feature alignment

            rt_loss = rt_loss_fn(global_map.view(-1, predictor.rooms), padded_gt_rt)

            savepath = './savi_rt_imgs/imgs11/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            if i % 10 == 1:
                heatmap_gt.save(savepath+"gt"+str(j)+"_"+str(i)+".jpg")
            if i % 10 == 1:
                heatmap_pred.save(savepath+"gmp"+str(j)+"_"+str(i)+".jpg")
            padded_gt_rt_view = padded_gt_rt.view(output_size, output_size)

    envs.close()
    
if __name__ == '__main__':
    train()