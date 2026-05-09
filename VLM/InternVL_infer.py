from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from torchvision import transforms as T

class InternVL:
    def __init__(self, version="2.5"):
        # Initialize InternVL2 model and tokenizer
        if version == "2.5_MPO":
            model_path = "OpenGVLab/InternVL2_5-8B-MPO"
        elif version == "2.5":
            model_path = "OpenGVLab/InternVL2_5-8B"
        elif version == "2":
            model_path = "OpenGVLab/InternVL2-8B"

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Helper function to preprocess images
    def build_transform(self, input_size):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    # Load and preprocess image using InternVL2's dynamic preprocess method
    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)

    def query_1(self, query_list):
        # Load and process the images
        pixel_values1 = self.load_image(query_list[0]).to(torch.bfloat16).cuda()
        pixel_values2 = self.load_image(query_list[1]).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        # Set generation parameters
        generation_config = dict(max_new_tokens=500, do_sample=False)

        # Describe images
        question = 'Image-1: <image>\nImage-2: <image>\n' + query_list[2]
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        # print("Response:", response)

        return response.split("\n")[-1].strip().lstrip()

    def query_2(self, query_list, summary_prompt, temperature=0.1):
        # Load and process the images
        pixel_values1 = self.load_image(query_list[0]).to(torch.bfloat16).cuda()
        pixel_values2 = self.load_image(query_list[1]).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        # Set generation parameters
        generation_config = dict(max_new_tokens=500, do_sample=False)

        # First stage: Describe images
        question = 'Image-1: <image>\nImage-2: <image>\n' + query_list[2]
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, return_history=True)

        # print("First stage result:", response)

        # Second stage: Summarize the first stage's result
        question = summary_prompt.format(response)
        response, history = self.model.chat(self.tokenizer, None, question, generation_config, history=history, return_history=True)

        # print("Second stage result:", response)

        return response.split("\n")[-1].strip().lstrip()