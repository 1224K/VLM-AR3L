from __future__ import annotations


import minedojo
from minedojo.sim.inventory import InventoryItem
import numpy as np

from .dense_reward import MobCombatDenseRewardWrapper

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(parent_dir)
import utils
from VLM import query_prompt
from VLM import phi

class CombatSpiderDenseRewardEnv(MobCombatDenseRewardWrapper):
    def __init__(
        self,
        step_penalty: float | int,
        attack_reward: float | int,
        success_reward: float | int,
        reward_mode: str="dense",
        reward_steps: int=1,
        reward_noise: float=0., 
    ):
        max_spawn_range = 10
        distance_to_axis = int(max_spawn_range / np.sqrt(2))
        spawn_range_low = (-distance_to_axis, 1, -distance_to_axis)
        spawn_range_high = (distance_to_axis, 1, distance_to_axis)

        env = minedojo.make(
            "Combat",
            target_names="spider",
            target_quantities=1,
            reward_weights=success_reward,
            # start_position=pos,
            initial_inventory=[
                InventoryItem(slot=0, name="diamond_sword", variant=None, quantity=1),
                InventoryItem(slot=36, name="diamond_boots", variant=None, quantity=1),
                InventoryItem(
                    slot=37, name="diamond_leggings", variant=None, quantity=1
                ),
                InventoryItem(
                    slot=38, name="diamond_chestplate", variant=None, quantity=1
                ),
                InventoryItem(slot=39, name="diamond_helmet", variant=None, quantity=1),
                InventoryItem(slot=40, name="shield", variant=None, quantity=1),
            ],
            initial_mobs="spider",
            initial_mob_spawn_range_low=spawn_range_low,
            initial_mob_spawn_range_high=spawn_range_high,
            image_size=(160, 256),
            world_seed=123,
            specified_biome="sunflower_plains",
            fast_reset=True,
            fast_reset_random_teleport_range=10,
            allow_mob_spawn=False,
            use_voxel=True,
            always_night=True,
            initial_weather="clear",
        )
        super().__init__(
            env=env,
            step_penalty=step_penalty,
            attack_reward=attack_reward,
        )

        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = 500
        self._elapsed_steps = 0
        self._first_reset = True

        self._reward_mode = reward_mode
        self._reward_steps = reward_steps
        self._reward_noise = reward_noise
        print(f"reward_mode: {self._reward_mode}, reward_steps: {self._reward_steps}, reward_noise: {self._reward_noise}")
        self._prev_image = None
        self.vlm_acc = 0
        self.vlm_cnt = 0

        if self._reward_mode=="phi":
            self.vlm = phi()

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(18000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False

        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True
        
        # dense reward
        if self._reward_mode=="dense":
            if self._elapsed_steps % self._reward_steps != 0:
                reward = 0
        # simple reward
        elif self._reward_mode=="simple":
            simple_reward = 0
            if self._elapsed_steps % self._reward_steps == 0:
                simple_reward = 0.1 if reward > 0 else 0
                # add noise
                if self._reward_noise > 0 and np.random.rand() < self._reward_noise:
                    simple_reward = 0 if simple_reward > 0 else 0.1
            reward = simple_reward
        # sparse reward
        elif self._reward_mode=="sparse":
            reward = 0 if reward < 10 else 10
        # phi reward
        elif self._reward_mode=="phi":
            vlm_reward = 0
            if self._elapsed_steps % self._reward_steps == 0:
                image = utils.obs_to_PIL_image(obs['rgb'])
                if self._prev_image is not None:
                    res = self.vlm.query_1([self._prev_image, image, query_prompt.format("combat a spider")])
                    if "1" in res:
                        vlm_reward = 0.1
                    else:
                        vlm_reward = 0
                    
                    # compute accuracy
                    if (vlm_reward > 0) == (reward > 0):
                        self.vlm_acc += 1
                    self.vlm_cnt += 1
                self._prev_image = image
            reward = vlm_reward 
        return obs, reward, done, info
