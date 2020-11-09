import json
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


# # Plot data
# for idx in range(50):
#     with open('bc_data/demo_{}.json'.format(idx)) as f:
#         demo = json.load(f)

#     obj_traj = np.array([item['obj_pos'] for item in demo])
#     obj_traj = obj_traj[::5]

#     plt.figure(0)
#     plt.plot(np.arange(len(obj_traj)), obj_traj[:, 0])

#     plt.figure(1)
#     plt.plot(np.arange(len(obj_traj)), obj_traj[:, 1])

#     plt.figure(2)
#     plt.plot(np.arange(len(obj_traj)), obj_traj[:, 2])

#     plt.figure(3)
#     plt.plot(obj_traj[:, 0], obj_traj[:, 2])

# plt.show()

np.random.seed(0)
all_idx = np.arange(50)
np.random.shuffle(all_idx)
train_idx = all_idx[:35]
val_idx = all_idx[35:45]
test_idx = all_idx[45:]

train_set = {'obj_pos': [], 'arm_action': [], 'gripper_action': []}
val_set = {'obj_pos': [], 'arm_action': [], 'gripper_action': []}
test_set = {'obj_pos': [], 'arm_action': [], 'gripper_action': []}

for idx in range(50):
    with open('bc_data/demo_{}.json'.format(idx)) as f:
        demo = json.load(f)

    obj_traj = np.array([item['obj_pos'] for item in demo])
    new_obj_traj = [obj_traj[0]]
    for i in range(1, len(obj_traj)):
        if np.linalg.norm(obj_traj[i] - new_obj_traj[-1]) > 0.02:
            new_obj_traj.append(obj_traj[i])
    new_obj_traj.append(obj_traj[-1])
    new_obj_traj = np.array(new_obj_traj)
    arm_action = np.array([new_obj_traj[i + 1] - new_obj_traj[i]
                           for i in range(len(new_obj_traj) - 1)])
    arm_action = np.vstack([arm_action, np.zeros(3)])
    gripper_action = np.zeros(len(arm_action))
    gripper_action[-1] = 1.0

    if idx in train_idx:
        dataset = train_set
    elif idx in val_idx:
        dataset = val_set
    else:
        dataset = test_set
    dataset['obj_pos'].extend(new_obj_traj.tolist())
    dataset['arm_action'].extend(arm_action.tolist())
    dataset['gripper_action'].extend(gripper_action.tolist())

with open('bc_data/train.json', 'w+') as f:
    json.dump(train_set, f)

with open('bc_data/val.json', 'w+') as f:
    json.dump(val_set, f)

with open('bc_data/test.json', 'w+') as f:
    json.dump(test_set, f)
