# Save job
```sh
scripts/save_job.sh k-minedojo
scripts/load_job.sh

scripts/save_job.sh k-deepseek
scripts/load_job.sh

scripts/save_job.sh k-metaworld
scripts/load_job.sh
```

# Connet VPN
```sh
source secrets/env.sh
scripts/vpn/disconnect.sh
scripts/vpn/connect.sh
```

## nuclus upload/download
- ### minedojo
    ```sh
    cd thirdparty/omnicli
    ./omnicli 
    auth omniverse nvidia
    copy "/home/mislab/Desktop/K/test_progressivity_mode/ppo_mineclip.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/ppo_mineclip.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/utils.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/utils.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/phi_infer.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/phi_infer.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/clip_infer.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/clip_infer.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/QwenVL_infer.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/QwenVL_infer.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/prompt.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/prompt.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/__init__.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/__init__.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/deepseekVL_infer.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/deepseekVL_infer.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/test_vlm.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/test_vlm.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/mob_combat/combat_spider.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/mob_combat/combat_spider.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/animal_zoo/milk_cow.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/animal_zoo/milk_cow.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/animal_zoo/hunt_cow.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/animal_zoo/hunt_cow.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/reward_model.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/reward_model.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/conv_net.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/conv_net.py"

    <!-- gif: combat_spider and hunt_cow,  model: all -->
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/gifs/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/gifs/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/logs/metaworld/drawer-open-v2/sac/sparse_n1_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/logs/metaworld/drawer-open-v2/sac/sparse_n1_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/tensorboard/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/tensorboard/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/model/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/model/minedojo/combat_spider/ppo/phi_n4_k16_noise0.0"
    cd ../..
    ```
- ### metaworld
    ```sh
    cd thirdparty/omnicli
    ./omnicli 
    auth omniverse nvidia
    copy "/home/mislab/Desktop/K/test_progressivity_mode/ppo_mineclip.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/ppo_mineclip.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/utils.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/utils.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/metaworld" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/metaworld"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/rlkit" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/rlkit"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/__init__.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/__init__.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/phi_infer.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/phi_infer.py"
    copy "/home/mislab/Desktop/K/test_progressivity_mode/VLM/prompt.py" "omniverse://nucleus.tpe1.local/Projects/K/minedojo/VLM/prompt.py"

    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/gifs/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/gifs/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/logs/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/logs/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/tensorboard/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/tensorboard/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0"
    copy "omniverse://nucleus.tpe1.local/Projects/K/minedojo/model/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0" "/home/mislab/Desktop/K/test_progressivity_mode/model/metaworld/soccer-v2/sac/phi_n16_k16_noise0.0"
    cd ../..
    ```

- ### RL-VLM-F
    ```sh
    cd thirdparty/omnicli
    ./omnicli 
    auth omniverse nvidia
    copy "/home/mislab/Desktop/K/RL-VLM-F" "omniverse://nucleus.tpe1.local/Projects/K/RL-VLM-F"

    copy "omniverse://nucleus.tpe1.local/Projects/K/RL-VLM-F/exp/gt_task_reward" "/home/mislab/Desktop/K/RL-VLM-F/exp/gt_task_reward"
    ```

# Copy
## nuclues to mnt
- ### minedojo
    ```sh
    scripts/submit_task.sh k-copy \
    "/run.sh \
        --download-src 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo' \
        --download-dest '/mnt/nfs/$FARM_USER/minedojo' \
        'cd /mnt/nfs/$FARM_USER/minedojo' \
        'ls' \
        'unzip attn.zip' \
        'rm -r tensorboard' \
        'rm -r logs' \
        'rm -r model' \
        'rm -r gifs' \
        'mkdir tensorboard' \
        'mkdir logs' \
        'mkdir model' \
        'mkdir gifs' \
        'cd ..' \
        'ls'" \
    "nuclues -> mnt: minedojo"
    ```

    ```sh
    scripts/submit_task.sh k-copy \
    "/run.sh \
        --download-src 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo/VLM/deepseekVL_infer.py' \
        --download-dest '/mnt/nfs/$FARM_USER/minedojo/VLM/deepseekVL_infer.py' \
        'cd /mnt/nfs/$FARM_USER/minedojo' \
        'ls'" \
    "nuclues -> mnt: minedojo"
    ```

- ### RL-VLM-F
    ```sh
    scripts/submit_task.sh k-copy \
    "/run.sh \
        --download-src 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/RL-VLM-F' \
        --download-dest '/mnt/nfs/$FARM_USER/RL-VLM-F' \
        'cd /mnt/nfs/$FARM_USER/RL-VLM-F' \
        'ls'" \
    "nuclues -> mnt: RL-VLM-F"
    ```

## mnt to nucleus
```sh
export TASK="combat_spider_phi_n16_noise0.0"
```

- gifs
    ```sh
    scripts/submit_task.sh  k-copy \
    "/run.sh \
        --upload-src '/mnt/nfs/$FARM_USER/minedojo/gifs/$TASK' \
        --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo/gifs/$TASK' \
        'ls /mnt/nfs/$FARM_USER/minedojo/gifs'" \
    "mnt -> nucleus: gifs"
    ```
- logs
    ```sh
    scripts/submit_task.sh  k-copy \
    "/run.sh \
        --upload-src '/mnt/nfs/$FARM_USER/minedojo/logs/$TASK' \
        --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo/logs/$TASK' \
        'ls /mnt/nfs/$FARM_USER/minedojo/logs/$TASK'" \
    "mnt -> nucleus: logs"
    ```
- tensorboard
    ```sh
    scripts/submit_task.sh  k-copy \
    "/run.sh \
        --upload-src '/mnt/nfs/$FARM_USER/minedojo/tensorboard/$TASK' \
        --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo/tensorboard/$TASK' \
        'ls /mnt/nfs/$FARM_USER/minedojo/tensorboard'" \
    "mnt -> nucleus: tensorboard"
    ```
- model
    ```sh
    scripts/submit_task.sh  k-copy \
    "/run.sh \
        --upload-src '/mnt/nfs/$FARM_USER/minedojo/model/$TASK' \
        --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo/model/$TASK' \
        'ls /mnt/nfs/$FARM_USER/minedojo/model'" \
    "mnt -> nucleus: model"
    ```
- RL-VLM-F
     ```sh
    scripts/submit_task.sh  k-copy \
    "/run.sh \
        --upload-src '/mnt/nfs/$FARM_USER/RL-VLM-F/exp' \
        --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/RL-VLM-F/exp' \
        'ls /mnt/nfs/$FARM_USER/RL-VLM-F/exp'" \
    "mnt -> nucleus: RL-VLM-F/exp"
    ```

# train
- ## minedojo
    ```sh
    scripts/submit_task.sh k-deepseek \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp  /mnt/nfs/$FARM_USER/minedojo/reward_model.py /src/reward_model.py' \
        'sudo cp  /mnt/nfs/$FARM_USER/minedojo/conv_net.py /src/conv_net.py' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user git+https://github.com/openai/CLIP.git --use-deprecated=legacy-resolver' \
        'xvfb-run python ppo_mineclip.py --mode train --env minedojo --task combat_spider --algo ppo --reward_mode deepseekVL --reward_steps 4 --reward_k 4 --seed 1'" \
    "combat_spider phi_n4_k4 seed1"
    ```

- ## metaworld
    ### drawer-open-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/metaworld /src/metaworld' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/rlkit /src/rlkit' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'xvfb-run python ppo_mineclip.py --mode train --env metaworld --task drawer-open-v2 --algo sac --reward_mode phi --reward_steps 4 --reward_k 4 --seed 3'" \
    "drawer-open-v2 phi --reward_steps 4 --reward_k 4 --seed 3"
    ```

    ### sweep-into-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/metaworld /src/metaworld' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/rlkit /src/rlkit' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'xvfb-run python ppo_mineclip.py --mode train --env metaworld --task sweep-into-v2 --algo sac --reward_mode phi --reward_steps 4 --reward_k 4 --seed 1'" \
    "sweep-into-v2 phi --reward_steps 4 --reward_k 4 --seed 1"
    ```
    
    ### soccer-v2
    ```sh 
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/metaworld /src/metaworld' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/rlkit /src/rlkit' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'xvfb-run python ppo_mineclip.py --mode train --env metaworld --task soccer-v2 --algo sac --reward_mode phi --reward_steps 4 --reward_k 4 --seed 3'" \
    "soccer-v2 phi --reward_steps 4 --reward_k 4 --seed 3"    
    ```

- ## RL-VLM-F
    ### phi
    #### drawer-open-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_drawer-open-v2 \
            seed=3 \
            exp_name=phi \
            reward=learn_from_preference \
            vlm_label=1 \
            vlm=phi \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=10 \
            num_interact=4000 \
            max_feedback=20000 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F drawer-open-v2 seed3"
    ```

    #### sweep-into-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_sweep-into-v2 \
            seed=3 \
            exp_name=phi \
            reward=learn_from_preference \
            vlm_label=1 \
            vlm=phi \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=10 \
            num_interact=4000 \
            max_feedback=20000 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F sweep-into-v2 seed3"
    ```

    #### soccer-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_soccer-v2 \
            seed=2 \
            exp_name=phi \
            reward=learn_from_preference \
            vlm_label=1 \
            vlm=phi \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=5 \
            num_interact=4000 \
            max_feedback=20000 \
            reward_lr=1e-4 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F soccer-v2 seed2"
    ```
    
    ### clip
    #### drawer-open-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_drawer-open-v2 \
            seed=3 \
            exp_name=clip \
            reward=clip_image_text_matching \
            vlm_label=1 \
            vlm=clip \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=10 \
            num_interact=4000 \
            max_feedback=20000 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F drawer-open-v2 seed=3"
    ```

    #### sweep-into-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_sweep-into-v2 \
            seed=3 \
            exp_name=gt_task_reward \
            reward=gt_task_reward \
            vlm_label=1 \
            vlm=clip \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=10 \
            num_interact=4000 \
            max_feedback=20000 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F sweep-into-v2 seed=3"
    ```

    #### soccer-v2
    ```sh
    scripts/submit_task.sh k-metaworld \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp -r /mnt/nfs/$FARM_USER/RL-VLM-F /src/RL-VLM-F' \
        'cd RL-VLM-F' \
        'ls' \
        'sudo rm -r /src/RL-VLM-F/exp' \
        'sudo ln -s /mnt/nfs/$FARM_USER/RL-VLM-F/exp /src/RL-VLM-F/exp' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'export GEMINI_API_KEY=K' \
        'export OPENAI_API_KEY=K' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user omegaconf==2.1.2 --use-deprecated=legacy-resolver' \
        'xvfb-run python train_PEBBLE.py \
            env=metaworld_soccer-v2 \
            seed=3 \
            exp_name=clip \
            reward=clip_image_text_matching \
            vlm_label=1 \
            vlm=clip \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=5 \
            num_interact=4000 \
            max_feedback=20000 \
            reward_lr=1e-4 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1'" \
    "RL-VLM-F soccer-v2 seed=3"
    ```

# eval
- ## minedojo
    ```sh
    scripts/submit_task.sh k-minedojo \
    "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp  /mnt/nfs/$FARM_USER/minedojo/reward_model.py /src/reward_model.py' \
        'sudo cp  /mnt/nfs/$FARM_USER/minedojo/conv_net.py /src/conv_net.py' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'sudo mkdir -p /home/ubuntu' \
        'sudo chmod -R 777 /home/ubuntu' \
        'sudo /opt/conda/bin/python3.9 -m pip install --no-user git+https://github.com/openai/CLIP.git --use-deprecated=legacy-resolver' \
        'xvfb-run python ppo_mineclip.py --mode eval --env minedojo --task milk_cow --algo ppo --reward_mode phi --reward_steps 4 --reward_k 16'" \
    "eval milk_cow phi_n4_k16"
    ```

- ## metaworld
    ```sh
    scripts/submit_task.sh k-metaworld \
        "/run.sh \
        'sudo mkdir -p /src' \
        'cd /src' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
        'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/metaworld /src/metaworld' \
        'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/rlkit /src/rlkit' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
        'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'sudo chmod -R 777 /src' \
        'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
        'xvfb-run python ppo_mineclip.py --mode eval --env metaworld --task sweep-into-v2 --algo sac --reward_mode dense'" \
    "eval sweep-into-v2 dense"
    ```

```sh
    xvfb-run python train_PEBBLE.py \
            env=metaworld_soccer-v2 \
            seed=2 \
            exp_name=clip \
            reward=learn_from_preference \
            vlm_label=1 \
            vlm=phi \
            image_reward=1 \
            reward_batch=40 \
            segment=1 \
            teacher_eps_mistake=0 \
            reward_update=5 \
            num_interact=4000 \
            max_feedback=20000 \
            reward_lr=1e-4 \
            agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
            num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
            diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
            feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
            num_eval_episodes=1 mode=eval agent_model_load_dir=/home/mislab/Desktop/K/RL-VLM-F/exp/phi/metaworld_soccer-v2/2025-01-18-09-07-47/vlm_1phi_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch40_Rupdate5_en3_sample0_large_batch10_seed2/models \
            reward_model_load_dir=/home/mislab/Desktop/K/RL-VLM-F/exp/phi/metaworld_soccer-v2/2025-01-18-09-07-47/vlm_1phi_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch40_Rupdate5_en3_sample0_large_batch10_seed2/models
```

# test_vlm
```sh
    scripts/submit_task.sh k-deepseek \
    "/run.sh \
        'mkdir -p /src' \
        'cd /src' \
        'cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
        'cp -r /mnt/nfs/$FARM_USER/minedojo/test_vlm.py /src/test_vlm.py' \
        'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
        'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
        'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
        'chmod -R 777 /src' \
        'chmod -R 777 /mnt/nfs/$FARM_USER' \
        'mkdir -p /home/ubuntu' \
        'source /opt/conda/bin/activate deepseekVL' \
        'python test_vlm.py --vlm deepseekVL2'" \
    "test_vlm.py --vlm deepseekVL2"
    ```