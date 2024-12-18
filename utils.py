import numpy as np
import torch
import PIL.Image as Image
def obs_to_PIL_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    return Image.fromarray(obs)
    