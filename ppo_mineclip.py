# gymnasium==0.28.1
# gym==0.21.0
import minedojo
import sys
import os
from animal_zoo import HuntCowDenseRewardEnv
from animal_zoo import MilkCowDenseRewardEnv
from mob_combat import CombatSpiderDenseRewardEnv
import numpy as np
from stable_baselines3 import PPO
from PIL import Image
import torch
import gym
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import imageio
from mineclip import MineCLIP
import matplotlib.pyplot as plt
import logging
import argparse

def transform_action_discrete(action):
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

def transform_action_multi_discrete(action):
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

def transform_action_multi_discrete2(action):
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

def obs_to_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    
    return obs

class MineCLIPFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, mineclip_model):
        super(MineCLIPFeatureWrapper, self).__init__(env)
        self.mineclip_model = mineclip_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_space = Box(
            low=-float('inf'), high=float('inf'), shape=(512,), dtype=np.float32
        )
        self.raw_rgb_obs = None

    def observation(self, obs):
        # Get the RGB observation
        self.raw_rgb_obs = obs['rgb']
        # Convert it to a tensor and normalize
        rgb_obs = torch.tensor(self.raw_rgb_obs.copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
        # Encode the observation using MineCLIP image_encoder
        image_feats = self.mineclip_model.forward_image_features(rgb_obs)
        return image_feats.squeeze(0).cpu().detach().numpy()

class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, transform_fn):
        super(CustomActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([12, 3])
        logger.info(f" Custom action space: {self.action_space}")
        self.transform_fn = transform_fn

    def action(self, action):
        # Apply custom transformation to the action
        return self.transform_fn(action)

class EvalCallbackWithGif(EvalCallback):
    def __init__(self, eval_env, n_eval_episodes, best_model_save_path, log_path, eval_freq, gif_path, **kwargs):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, **kwargs)
        self.gif_path = gif_path
        os.makedirs(gif_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            gif_filename = os.path.join(self.gif_path, f"eval_{self.n_calls * 4}.gif") # n_envs = 4
            self._generate_gif(gif_filename)
        return result
    
    def _generate_gif(self, filename):
        images = []
        obs = self.eval_env.reset()
        done = False
        while not done:
            # Predict the action for the current observation
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)

            # Capture the raw RGB observation for GIF
            try:
                current_obs = self.eval_env.get_attr('raw_rgb_obs')[0]
            except:
                current_obs = obs  # Adjust this depending on your environment structure

            images.append(self._obs_to_image(current_obs))
        
        # Save the images as a GIF
        imageio.mimsave(filename, images, fps=10)

    def _obs_to_image(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        obs = np.transpose(obs, (1, 2, 0))
        obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
        obs = obs.astype(np.uint8)
        return obs
    
def train_and_evaluate(mode, task, reward_mode, reward_steps, reward_noise, seed):
    if "combat_spider" in task:
        if "train" in mode:
            train_env = CombatSpiderDenseRewardEnv(
                step_penalty=0,
                attack_reward=1,
                success_reward=10,
                reward_mode = reward_mode,
                reward_steps = reward_steps,
                reward_noise = reward_noise,
            )
        eval_env = CombatSpiderDenseRewardEnv(
            step_penalty=0,
            attack_reward=1,
            success_reward=10,
        )
    elif "milk_cow" in task:
        if "train" in mode:
            train_env = MilkCowDenseRewardEnv(
                step_penalty=0,
                nav_reward_scale=0.1,
                success_reward=10,
                reward_mode = reward_mode,
                reward_steps = reward_steps,
                reward_noise = reward_noise,
            )
        eval_env = MilkCowDenseRewardEnv(
            step_penalty=0,
            nav_reward_scale=0.1,
            success_reward=10,
        )
    elif "hunt_cow" in task:
        if "train" in mode:
            train_env = HuntCowDenseRewardEnv(
                step_penalty=0,
                nav_reward_scale=0.1,
                attack_reward=1,
                success_reward=10,
                reward_mode = reward_mode,
                reward_steps = reward_steps,
                reward_noise = reward_noise,
            )
        eval_env = HuntCowDenseRewardEnv(
            step_penalty=0,
            nav_reward_scale=0.1,
            attack_reward=1,
            success_reward=10,
        )
    
    # Load the MineCLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mineclip_model = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        resolution=(160, 256),
        pool_type="attn.d2.nh8.glusw",
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        hidden_dim=512,
    ).to(device)
    mineclip_model.load_ckpt("attn.pth")
    logger.info(f" Load MineCLIP model")

    # Define the action transformation function
    transform = transform_action_multi_discrete2

    name = task + "_" + reward_mode + "_n" + str(reward_steps) + "_noise" + str(reward_noise)
    print(f"[INFO] model: {name}")

    # Define the evaluation callback
    if "random" not in mode:
        eval_env = MineCLIPFeatureWrapper(eval_env, mineclip_model)
        eval_env = CustomActionWrapper(eval_env, transform)
        eval_env = make_vec_env(lambda: eval_env, n_envs=1)
        eval_env = VecFrameStack(eval_env, n_stack=4)

    if "train" in mode:
        train_env = MineCLIPFeatureWrapper(train_env, mineclip_model)
        train_env = CustomActionWrapper(train_env, transform)
        train_env = make_vec_env(lambda: train_env, n_envs=4)
        train_env = VecFrameStack(train_env, n_stack=4)

        # tensorboard writer
        log_dir = f"./tensorboard/{name}"

        # Create the evaluation callback
        eval_callback = EvalCallbackWithGif(
            eval_env=eval_env,
            n_eval_episodes=5,
            best_model_save_path=f'./model/best_model/{name}/{seed}',
            log_path=f'./logs_result/{name}/{seed}',
            eval_freq=5120,  # n_envs * n_steps = 4 * 5120 = 20480
            gif_path=f'./gifs/{name}/{seed}',
            deterministic=True,
            render=False,
        )

    if mode=='noop':
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        t = 0
        while not done:
            action = eval_env.action_space.no_op()
            if t == 0:
                action[3] = 13
            else:
                action[3] = 12
            t = 1
            print(action)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            rgb_image = obs_to_image(obs)
            images.append(rgb_image)

        print(f"Total reward: {total_reward}")
    elif "random" in mode:
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        while not done:
            action = eval_env.action_space.sample()
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            current_obs = eval_env.raw_rgb_obs
            rgb_image = obs_to_image(current_obs)
            images.append(rgb_image)

        print(f"Total reward: {total_reward}")
        imageio.mimsave(f"gifs/random{mode[6:]}.gif", images, fps=10)
    elif "eval" in mode:
        # check if the model exists
        if not os.path.exists(f"model/best_model/{name}/{seed}/best_model.zip"):
            print(f"The model model/best_model/{name}/{seed}/best_model.zip does not exist.")
            return [], []
        
        model = PPO.load(f"model/best_model/{name}/{seed}/best_model", env=eval_env)
        # print(model.policy)
        average_success_rate = []
        reward_list = []
        for i in range(20):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            images = []
            while not done:
                action, _ = model.predict(obs.copy())
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
                # use non-stacked observation to save gif
                current_obs = eval_env.get_attr('raw_rgb_obs')[0]
                rgb_image = obs_to_image(current_obs)
                images.append(rgb_image)

            print(f"Total reward: {total_reward}")
            imageio.mimsave(f"gifs/{name}/{seed}/eval-{i}.gif", images, fps=60)
            average_success_rate.extend(total_reward >= 10)
            reward_list.append(total_reward)

        eval_env.close()
        return average_success_rate, reward_list
    elif "train" in mode:
        model = PPO("MlpPolicy", train_env, ent_coef=0.01, verbose=1, tensorboard_log=log_dir, seed=seed)
        model.learn(total_timesteps=1000000, callback=eval_callback)
        model.save("./model/best_model/"+ name + f"/{seed}")
        if reward_mode == "phi":
            # save the accuracy
            if not os.path.exists(f"./logs_result/{name}/{seed}"):
                os.makedirs(f"./logs_result/{name}/{seed}")
            with open(f"./logs_result/{name}/{seed}/vlm_accuracy.txt", "w") as f:
                total_acc = 0
                total_cnt = 0
                for i in range(4):
                    total_acc += train_env.get_attr('vlm_acc')[i]
                    total_cnt += train_env.get_attr('vlm_cnt')[i]
                if total_cnt > 0:
                    f.write(f"vlm_accuracy: {total_acc / total_cnt}")
        train_env.close()
        eval_env.close()
        torch.cuda.empty_cache()

    elif mode=='gen_data':
        model =  PPO.load(f"model/best_model/{name}/{seed}/best_model", env=eval_env)
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        rewards = []
        step = 0
        n_step = 16
        image_path = f"./VLM/data/{task}/"
        label0_path = image_path + "0/"
        label1_path = image_path + "1/"
        os.makedirs(label0_path, exist_ok=True)
        os.makedirs(label1_path, exist_ok=True)
        label0_count = len(os.listdir(label0_path)) / 2
        label1_count = len(os.listdir(label1_path)) / 2
        print("original label 0 count: ", label0_count)
        print("original label 1 count: ", label1_count)
        # rename the images
        i0 = 0 
        i = 0
        while i0 < label0_count:
            if os.path.exists(label0_path + str(int(i)) + "0.png"):
                os.rename(label0_path + str(int(i)) + "0.png", label0_path + str(int(i0)) + "0.png")
                os.rename(label0_path + str(int(i)) + "1.png", label0_path + str(int(i0)) + "1.png")
                i0 += 1
            i += 1
        i1 = 0
        i = 0
        while i1 < label1_count:
            if os.path.exists(label1_path + str(int(i)) + "0.png"):
                os.rename(label1_path + str(int(i)) + "0.png", label1_path + str(int(i1)) + "0.png")
                os.rename(label1_path + str(int(i)) + "1.png", label1_path + str(int(i1)) + "1.png")
                i1 += 1
            i += 1

        # 21
        while label1_count < 50:
            obs = eval_env.reset()
            done = False
            step = 0
            while not done:
                action, _ = model.predict(obs.copy())
                obs, reward, done, info = eval_env.step(action)
                current_obs = eval_env.get_attr('raw_rgb_obs')[0]
                rgb_image = obs_to_image(current_obs)
                images.append(rgb_image)
                rewards.append(reward)
                if step >= n_step and (step % 5 == 0 or reward >=1):
                    if rewards[step] > rewards[step-n_step] and label1_count < 50:
                        # save images[step] and images[step-n_step] to label 1
                        image = Image.fromarray(images[step-n_step])
                        image.save(label1_path + str(int(label1_count)) + "0.png")
                        image = Image.fromarray(images[step])
                        image.save(label1_path + str(int(label1_count)) + "1.png")
                        label1_count += 1
                    elif rewards[step] <= rewards[step-n_step] and label0_count < 50:
                        # save images[step] and images[step-n_step] to label 0
                        image = Image.fromarray(images[step-n_step])
                        image.save(label0_path + str(int(label0_count)) + "0.png")
                        image = Image.fromarray(images[step])
                        image.save(label0_path + str(int(label0_count)) + "1.png")
                        label0_count += 1
                step += 1
        

def draw(name, task, exp, max_timestep=1000000, sigma=1):
    mean_rewards = {}
    se_rewards = {}
    
    for exp_name in exp:
        if not os.path.exists(f'logs/{task}_' + exp_name):
            print(f"[ERROR] The experiment {task}_{exp_name} does not exist.")
            return
        rewards_timestep = {}
        for file in os.listdir(f'logs/{task}_' + exp_name):
            if os.path.isdir(f'logs/{task}_{exp_name}/{file}') != True:
                continue
            data = np.load(f'logs/{task}_{exp_name}/{file}/evaluations.npz')
            print(f"[INFO] Load data from logs/{task}_{exp_name}/{file}/evaluations.npz")
            reward = data['results']
            timestep = data['timesteps']
            for i in range(len(reward)):
                if timestep[i] > max_timestep:
                    break
                if timestep[i] not in rewards_timestep:
                    rewards_timestep[timestep[i]] = []
                rewards_timestep[timestep[i]].append(np.mean(reward[i]))

        mean_rewards[exp_name] = {t: np.mean(rewards_timestep[t]) for t in rewards_timestep}
        se_rewards[exp_name] = {t: np.std(rewards_timestep[t]) / np.sqrt(len(rewards_timestep[t])) for t in rewards_timestep}
    
    # colors = [plt.cm.magma(i) for i in np.linspace(0, 1, len(exp))]
    # colors = [plt.cm.Set2(i) for i in range(len(exp))]
    # colors = ['C5', 'C3', 'C4', 'C0', 'C6', 'C9', 'C1']
    colors = ['C5', 'C0', 'C6', 'C9', 'C1']
    for exp_name in exp:
        label_name = exp_name
        if "_noise0.0" in exp_name:
            label_name = exp_name.replace("_noise0.0", "")
        # if "_n1_" in exp_name:
        #     label_name = label_name.replace("_n1", "")
        # if "simple_n1_" in exp_name:
        #     label_name = "oracle"
        from scipy.ndimage import gaussian_filter1d
        smooth_mean = gaussian_filter1d(list(mean_rewards[exp_name].values()), sigma=sigma)
        smooth_se = gaussian_filter1d(list(se_rewards[exp_name].values()), sigma=sigma)
        if "noise" in label_name:
            plt.plot(list(mean_rewards[exp_name].keys()), smooth_mean, label=label_name, color=colors[exp.index(exp_name)%len(colors)], linestyle="--")
        else:
            plt.plot(list(mean_rewards[exp_name].keys()), smooth_mean, label=label_name, color=colors[exp.index(exp_name)%len(colors)])
        plt.fill_between(list(mean_rewards[exp_name].keys()), smooth_mean - smooth_se, smooth_mean + smooth_se, color=colors[exp.index(exp_name)%len(colors)], alpha=0.3)

    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.ylim(0, 11) 
    plt.title(task.replace("_", " ").capitalize())
    plt.legend()
    # plt.legend(ncol=2)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # save the figure
    if not os.path.exists('plot/' + name):
        os.makedirs('plot/' + name)
    plt.savefig(f"plot/{name}/mean_reward_smooth{sigma}.png")
    # plt.savefig(f"plot/{name}/mean_reward_smooth{sigma}.png", bbox_inches='tight')
    print(f"[INFO] Save the plot to plot/{name}/mean_reward_smooth{sigma}.png")

def parse_range_or_list(value):
    """
    Parses a string representing a range (e.g., "1-5") or a list (e.g., "1,2,3").
    Returns a list of integers.
    """
    if "-" in value:  # Range format
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    elif "," in value:  # List format
        return list(map(int, value.split(",")))
    else:
        # Single integer
        return [int(value)]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='ERROR') # DEBUG, INFO, WARNING, ERROR, CRITICAL
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        help="Mode of operation: e.g., 'train', 'eval', etc."
    )
    parser.add_argument(
        '--task', 
        type=str, 
        required=True, 
        help="task: e.g., 'hunt_cow', 'combat_spider'"
    )
    parser.add_argument(
        '--reward_mode', 
        type=str, 
        default='dense',
        help="reward_mode: e.g., 'dense', 'simple', 'phi'"
    )
    parser.add_argument(
        '--reward_steps', 
        type=int, 
        default=1,
        help="reward_steps: e.g., 1, 2, 4 ..."
    )
    parser.add_argument(
        '--reward_noise', 
        type=float, 
        default=0.,
        help="reward_noise: e.g., 0., 0.3, ..."
    )
    parser.add_argument(
        '--seed', 
        type=parse_range_or_list, 
        default=[1, 2, 3],
        help="random seed, e.g., '1,2,3', '1-3'"
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)

    mode = args.mode
    task = args.task
    reward_mode = args.reward_mode
    reward_steps = args.reward_steps
    reward_noise = args.reward_noise
    seed_list = args.seed
    
    # ln -s ./logs ./logs_result
    # xvfb-run python ppo_mineclip.py --mode train --task milk_cow --reward_mode phi --reward_steps 16 --seed 3
    if mode == 'train':
        for seed in seed_list:
            train_and_evaluate(mode, task, reward_mode, reward_steps, reward_noise, seed)
    # xvfb-run python ppo_mineclip.py --mode eval --task hunt_cow --reward_mode phi --reward_steps 16 --reward_noise 0 
    elif mode == 'eval':
        average_success_rate = []
        average_reward = []
        with open(f"./logs_result/{task}_{reward_mode}_n{str(reward_steps)}_noise{str(reward_noise)}/success_rate.txt", "a") as f:
            for seed in seed_list:
                success_rate, reward = train_and_evaluate(mode, task, reward_mode, reward_steps, reward_noise, seed)
                if len(success_rate) == 0:
                    continue
                mean_success_rate = np.mean(success_rate)
                mean_reward = np.mean(reward)
                print(f"seed {seed} success rate: {mean_success_rate}, reward: {mean_reward}")
                f.write(f"seed {seed} success rate: {mean_success_rate}, reward: {mean_reward}\n")
                for i in range(len(reward)):
                    f.write(f"reward {i}: {reward[i]}\n")
                average_success_rate.append(mean_success_rate)
                average_reward.append(mean_reward)
            print(f"success average: {np.mean(average_success_rate) * 100} SE: {np.std(average_success_rate)/ np.sqrt(len(average_success_rate)) * 100}")
            print(f"reward average: {np.mean(average_reward)} SE: {np.std(average_reward)/ np.sqrt(len(average_reward))}")
            f.write(f"success average: {np.mean(average_success_rate) * 100} SE: {np.std(average_success_rate)/ np.sqrt(len(average_success_rate)) * 100}\n")
            f.write(f"reward average: {np.mean(average_reward)} SE: {np.std(average_reward)/ np.sqrt(len(average_reward))}\n")
    # python ppo_mineclip.py --mode draw --task hunt_cow
    elif mode == 'draw':
        # draw(f'{task}/compare', task, ['dense_n1_noise0.0', 'sparse_n1_noise0.0', 'simple_n1_noise0.0', 'phi_n4_noise0.0', 'phi_n16_noise0.0'], sigma=2)
        draw(f'{task}/dense', task, ['dense_n1_noise0.0', 'dense_n2_noise0.0',
                                                'dense_n4_noise0.0', 'dense_n8_noise0.0', 'dense_n16_noise0.0'] ,sigma=2)
        # draw(f'{task}/simple_noise', task, [
        #     'simple_n1_noise0.0', 'simple_n2_noise0.0',
        #                                     'simple_n4_noise0.0', 'simple_n8_noise0.0', 'simple_n16_noise0.0',
        #     'simple_n1_noise0.3', 'simple_n2_noise0.3',
        #                                     'simple_n4_noise0.3', 'simple_n8_noise0.3', 'simple_n16_noise0.3',
        #                                     ], sigma=2)
    # xvfb-run python ppo_mineclip.py --mode gen_data --task combat_spider --reward_mode dense 
    elif mode == 'gen_data':
        train_and_evaluate(mode, task, reward_mode, reward_steps, reward_noise, 1)
