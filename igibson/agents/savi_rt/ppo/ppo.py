#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from igibson.agents.savi_rt.utils.utils import to_tensor


EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts, loss_weight=None):
        advantages = self.get_advantages(rollouts)

        num_epoch = 5
        num_mini_batch = 1

        rt_num_correct = 0
        rt_num_sample = 0

        value_loss_epoch = 0
        rt_value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    external_memory,
                    external_memory_masks,
                ) = sample
                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    rt_map,
                    _,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                    external_memory,
                    external_memory_masks,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                # origin rt map shape (batch, 23, 50, 50)
                # permute to (batch, 50, 50, 23)
                rt_map = rt_map.permute(0, 2, 3, 1).contiguous().view(rt_map.shape[0], -1, 23).contiguous()
                # reshape to (batch*50*50, 23)
                rt_map = rt_map.reshape(-1, 23)
                # ground truth of rt map (batch, 50, 50), reshape to (batch, 50*50)
                rt_map_gt = to_tensor(obs_batch['rt_map_gt']).view(rt_map.shape[0], -1).contiguous().to(self.device)
                # again reshape tp (batch*50*50)
                rt_map_gt = rt_map_gt.reshape(-1)
                # pass rt map and gt to the NonZeroWeightedCrossEntropy
                rt_loss = self.actor_critic.net.rt_loss_fn_class(rt_map, rt_map_gt)
                

                # accumulate the rt loss
                rt_value_loss_epoch += rt_loss.item()
                # get the prediction from (batch*32*32, 23) to (batch*32*32,)
                rt_preds = torch.argmax(rt_map, dim=1)
                # accumulate the correctly predicted pixel
                rt_num_correct += torch.sum(torch.eq(rt_preds, rt_map_gt))
                # accumulate the total pixels
                rt_num_sample += rt_map_gt.shape[0]

                if rt_loss is not None:
                    total_loss = self.actor_critic.net.policy_loss_weight * total_loss + loss_weight*rt_loss

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        rt_value_loss_epoch /= num_epoch * num_mini_batch
        rt_num_correct = rt_num_correct.item() / rt_num_sample

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, rt_value_loss_epoch, rt_num_correct

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass

