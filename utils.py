import numpy as np
import torch
import PIL.Image as Image
from gym.wrappers.time_limit import TimeLimit
import gym
from gym.spaces import Box
from VLM import task_prompt, query_prompt, clip_env_prompts
from VLM import phi
from VLM import clip_infer_score
def obs_to_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    
    return obs

def obs_to_PIL_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    return Image.fromarray(obs)
    
class ClipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip_range=(-1e6, 1e6)):
        super().__init__(env)
        self.clip_low, self.clip_high = clip_range
        low = np.clip(env.observation_space.low, self.clip_low, self.clip_high)
        high = np.clip(env.observation_space.high, self.clip_low, self.clip_high)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        if isinstance(observation, tuple): # for metaworld envs ingore reset info 
            return observation[0]
        return observation
    
def make_metaworld_env(task, seed):
    import metaworld.envs.mujoco.env_dict as _env_dict
    from rlkit.envs.wrappers import NormalizedBoxEnv

    if task in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[task]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[task]
    
    env = env_cls(render_mode='rgb_array')
    env.camera_name = task
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    env = ClipObservationWrapper(env)

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

def make_minedojo_env(task):
    from animal_zoo import HuntCowDenseRewardEnv
    from animal_zoo import MilkCowDenseRewardEnv
    from mob_combat import CombatSpiderDenseRewardEnv
    if task == "combat_spider":
        env = CombatSpiderDenseRewardEnv(step_penalty=0, attack_reward=1, success_reward=10)
    elif task == "milk_cow":
        env = MilkCowDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, success_reward=10)
    elif task == "hunt_cow":
        env = HuntCowDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, attack_reward=1, success_reward=10)
    else:
        raise NotImplementedError
    return env
    
class make_reward_env(gym.Wrapper):
    def __init__(self, env, env_name, task, reward_mode, reward_steps:int=1, reward_noise:float=0., reward_k:int=16, reward_model=None):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        self._steps = 0

        self._task = task
        self._reward_mode = reward_mode
        self._reward_steps = reward_steps
        self._reward_noise = reward_noise
        self._reward_k = reward_k
        self.reward_model = reward_model
        print(f"reward_mode: {self._reward_mode}, reward_steps: {self._reward_steps}, reward_noise: {self._reward_noise}, reward_k: {self._reward_k}")
        self._prev_image = [None] * self._reward_k
        self._prev_reward = [0] * self._reward_k
        self.vlm_acc = 0
        self.vlm_cnt = 0

        if self._reward_mode=="phi":
            self.vlm = phi()

    def reset(self, **kwargs):
        self._steps = 0

        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1
        
        # dense reward
        if self._reward_mode=="dense":
            if self._steps % self._reward_steps != 0:
                reward = 0
        # simple reward (oracle)
        elif self._reward_mode=="simple":
            simple_reward = 0
            if self._steps % self._reward_steps == 0:
                simple_reward = 0.1 if reward > 0 else 0
                # add noise
                if self._reward_noise > 0 and np.random.rand() < self._reward_noise:
                    simple_reward = 0 if simple_reward > 0 else 0.1
            reward = simple_reward
        # sparse reward
        elif self._reward_mode=="sparse":
            if self.env_name == "metaworld":
                reward = info["success"]
            elif self.env_name == "minedojo":
                reward = 0 if reward < 10 else 10
            else:
                raise NotImplementedError
        # VLM reward
        elif self._reward_mode=="phi":
            vlm_reward = 0
            idx = self._steps % self._reward_k
            
            # get image
            if self.env_name == "metaworld":
                image = obs_to_PIL_image(obs['image'])
            elif self.env_name == "minedojo":
                image = obs_to_PIL_image(obs['rgb'])
            else:
                raise NotImplementedError
            
            if self._steps % self._reward_steps == 0:
                if self._prev_image[idx] is not None:
                    res = self.vlm.query_1([self._prev_image[idx], image, query_prompt.format(task_prompt[self._task])])
                    if "1" in res:
                        vlm_reward = 0.1
                    else:
                        vlm_reward = 0
                    
                    # compute accuracy
                    if (vlm_reward > 0) == (reward > self._prev_reward[idx]):
                        self.vlm_acc += 1
                    self.vlm_cnt += 1
                self._prev_image[idx] = image
                self._prev_reward[idx] = reward
            reward = vlm_reward
        # RL-VLM-F
        elif self._reward_mode=="RL-VLM-F":
            if self.env_name == "minedojo":
                rgb_image = obs['rgb'].transpose(1, 2, 0) # (H, W, C)
                image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # 轉置為 (C, H, W)
                image = image.reshape(1, 3, image.shape[1], image.shape[2]) # 增加 batch 維度
            else:
                raise NotImplementedError
            
            self.reward_model.add_data(obs['rgb'].flatten(), action, reward, done, img=rgb_image)
            self.reward_model.eval()
            reward = self.reward_model.r_hat(image)
            self.reward_model.train()
        # clip similarity score
        elif self._reward_mode=="clip":
            if self.env_name == "minedojo":
                rgb_image = obs['rgb'].transpose(1, 2, 0) # (H, W, C)
            else:
                raise NotImplementedError
            clip_infer_score(rgb_image, clip_env_prompts[self._task]) * 2 - 1 # actually we should scale it [-1, 1] since tanh is used in the reward model
        else:
            raise NotImplementedError
        
        return obs, reward, done, info
    
    def get_attr(self, str):
        if str == 'vlm_acc':
            return self.vlm_acc
        elif str == 'vlm_cnt':
            return self.vlm_cnt
        
def concatenate_images_vertical(images, dist_images):
    # calc max width from imgs
    width = max(img.width for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.height for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    # init var to track current height pos
    current_height = 0
    for img in images:
        # paste img in new_img at current height
        new_img.paste(img, (0, current_height))
        # update current height for next img
        current_height += img.height + dist_images

    return new_img

def minedojo_transform_action_discrete(action):
    """
    Map agent action to env action.
    """
    # discrete(17) -> multiDiscrete[3, 3, 4, 25, 25, 8, 244, 36]
    # 3x3 camera + 6 action + 2 function 
    # action: forward, forward + jump, jump, back, move left, and move right
    # function: use, attack
    action_t = []
    if action == 9 or action == 10: # forward, forward + jump
        action_t.append(1)
    elif action == 12: # back
        action_t.append(2)
    else:
        action_t.append(0)

    if action == 13: # move left
        action_t.append(1)
    elif action == 14: # move right
        action_t.append(2)
    else:
        action_t.append(0)
    
    if action == 10 or action == 11: # forward + jump, jump
        action_t.append(1)
    else:
        action_t.append(0)
    
    if action < 9:
        pitch = action // 3
        yaw = action % 3
        action_t.append(pitch + 11) # 11 ~ 13
        action_t.append(yaw + 11) # 11 ~ 13
    else:
        action_t.append(12)
        action_t.append(12)
    
    if action == 15: # use
        action_t.append(1)
    elif action == 16: # attack
        action_t.append(3)
    else:
        action_t.append(0)
    
    action_t.append(0)
    action_t.append(0)
    return action_t

def minedojo_transform_action_multi_discrete(action):
    """
    Map agent action to env action.
    """
    # MultiDiscrete([3, 3, 4, 3, 3, 3]) -> multiDiscrete[3, 3, 4, 25, 25, 8, 244, 36]
    action_t = action.copy()
    action_t[3] += int(11)
    action_t[4] += int(11)
    if action_t[5] == 2:
        action_t[5] = 3
    action_t = np.concatenate([action_t, np.array([0, 0])])
    return action_t

def minedojo_transform_action_multi_discrete2(action):
    """
    Map agent action to env action.
    """
    # MultiDiscrete([12, 3]) -> multiDiscrete[3, 3, 4, 25, 25, 8, 244, 36]
    # first 12 actions: NO_OP, forward, backward, left, right, jump, sneak, sprint, camera pitch +30, camera pitch -30, camera yaw +30, and camera yaw -30.
    # last 3 actions: NO_OP, use and attack
    action_t = []
    if action[0] == 1: # forward
        action_t.append(1)
    elif action[0] == 2: # backward
        action_t.append(2)
    else:
        action_t.append(0)

    if action[0] == 3: # left
        action_t.append(1)
    elif action[0] == 4: # right
        action_t.append(2)
    else:
        action_t.append(0)

    if action[0] == 5: # jump
        action_t.append(1)
    elif action[0] == 6: # sneak
        action_t.append(2)
    elif action[0] == 7: # sprint
        action_t.append(3)
    else:
        action_t.append(0)

    if action[0] == 8: # camera pitch +30
        action_t.append(11)
    elif action[0] == 9: # camera pitch -30
        action_t.append(13)
    else:
        action_t.append(12)

    if action[0] == 10: # camera yaw +30
        action_t.append(11)
    elif action[0] == 11: # camera yaw -30
        action_t.append(13)
    else:
        action_t.append(12)

    if action[1] == 1: # use
        action_t.append(1)
    elif action[1] == 2: # attack
        action_t.append(3)
    else:
        action_t.append(0)

    action_t.append(0)
    action_t.append(0)
    return action_t