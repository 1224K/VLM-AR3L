import torch
from PIL import Image
import torchvision.transforms as transforms

class mineclip:
    def __init__(self):
        from mineclip import MineCLIP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vlm = MineCLIP(
                arch="vit_base_p16_fz.v2.t2",
                resolution=(160, 256),
                pool_type="attn.d2.nh8.glusw",
                image_feature_dim=512,
                mlp_adapter_spec="v0-2.t0",
                hidden_dim=512,
            ).to(self.device)
        
    def query_1(self, image1, image2, query_prompt):
        # image to tensor (1, 1, C, H, W)
        image1_tensor = transforms.ToTensor()(image1).unsqueeze(0).unsqueeze(0).to(self.device)
        image2_tensor = transforms.ToTensor()(image2).unsqueeze(0).unsqueeze(0).to(self.device)
        image1_feats = self.vlm.forward_image_features(image1_tensor)
        video1_feats = self.vlm.forward_video_features(image1_feats)
        reward1 = self.vlm.forward_reward_head(video1_feats, text_tokens=query_prompt)[0].item()
        image2_feats = self.vlm.forward_image_features(image2_tensor)
        video2_feats = self.vlm.forward_video_features(image2_feats)
        reward2 = self.vlm.forward_reward_head(video2_feats, text_tokens=query_prompt)[0].item()
        
        if reward2 > reward1:
            return "1"
        else:
            return "0"