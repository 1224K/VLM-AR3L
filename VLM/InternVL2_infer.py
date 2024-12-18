from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time
import torch
import os
import sys
from torchvision import transforms as T

# Initialize InternVL2 model and tokenizer
model_path = "OpenGVLab/InternVL2-8B"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Helper function to preprocess images
def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
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
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

def InternVL2_query_1(query_list, temperature=0.1):
    beg = time.time()

    # Load and process the images
    pixel_values1 = load_image(query_list[0]).to(torch.bfloat16).cuda()
    pixel_values2 = load_image(query_list[1]).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # Set generation parameters
    generation_config = dict(max_new_tokens=500, do_sample=False)

    # Describe images
    question = 'Image-1: <image>\nImage-2: <image>\n' + query_list[2]
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print("Response:", response)

    end = time.time()
    print("Time elapsed:", end - beg)

    return response.split("\n")[-1].strip().lstrip()

def InternVL2_query_2(query_list, summary_prompt, temperature=0.1):
    beg = time.time()

    # Load and process the images
    pixel_values1 = load_image(query_list[0]).to(torch.bfloat16).cuda()
    pixel_values2 = load_image(query_list[1]).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # Set generation parameters
    generation_config = dict(max_new_tokens=500, do_sample=False)

    # First stage: Describe images
    question = 'Image-1: <image>\nImage-2: <image>\n' + query_list[2]
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, return_history=True)

    print("First stage result:", response)

    # Second stage: Summarize the first stage's result
    question = summary_prompt.format(response)
    response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)

    print("Second stage result:", response)

    end = time.time()
    print("Time elapsed:", end - beg)

    return response.split("\n")[-1].strip().lstrip()

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    from prompt import image_0_prompt, image_1_prompt, query_prompt, thought_prompt, summary_prompt, query_CoT_prompt

    image_dir = "./data/"

    table_acc = {}
    table_step = {}

    for env_name in os.listdir(image_dir):
        if os.path.isdir(image_dir + env_name) == False:
            continue
        print("env_name:", env_name)
        acc = {0:0, 1:0}
        step = 0
        image_path = image_dir + env_name + "/"
        for label in os.listdir(image_path):
            if (label == "0" or label == "1") == False:
                continue
            label_i = int(label)
            i = 0
            while os.path.exists(image_path + label + "/" + str(i) + "0.png"):
                image_0_path = image_path + label + "/" + str(i) + "0.png"
                image_1_path = image_path + label + "/" + str(i) + "1.png" 
                print("\n\nImage:", i)

                i += 1
                try:
                    image_0 = Image.open(image_0_path)
                    image_1 = Image.open(image_1_path)
                except:
                    print("image not found")
                    continue

                for _ in range(1):
                    # ans = InternVL2_query_1(
                    #     [
                    #         image_0_path,
                    #         image_1_path,
                    #         query_prompt.format("hunt a cow"),
                    #     ],
                    #     )
                    # ans = InternVL2_query_1(
                    #     [
                    #         image_0_path,
                    #         image_1_path,
                    #         query_CoT_prompt.format("hunt a cow"),
                    #     ],
                    # )
                    ans = InternVL2_query_2(
                        [
                            image_0_path,
                            image_1_path,
                            thought_prompt.format("hunt a cow"),
                        ],
                        summary_prompt.format("hunt a cow", "{}"),
                        )


                    if "0" in ans:
                        ans = 0
                    elif "1" in ans:
                        ans = 1
                    else:
                        ans = 0
                    print("ans: ", ans, "| ground truth: ", label_i)

                    if ans == label_i:
                        acc[ans] += 1
                    step += 1
            if table_acc.get(env_name) == None:
                table_acc[env_name] = {}
                table_step[env_name] = {}
            table_acc[env_name][label_i] = acc[label_i]
            table_step[env_name][label_i] = i
    total_acc = 0
    total_step = 0
    for env_name in table_acc:
        env_acc = 0
        env_step = 0
        for label in table_acc[env_name]:
            env_acc += table_acc[env_name][label]
            env_step += table_step[env_name][label]
            print("env_name: ", env_name, "| label: ", label, "| acc: ", table_acc[env_name][label] * 1.0 / table_step[env_name][label], "| ac ", table_acc[env_name][label] , "| step: ", table_step[env_name][label])
        print("env_name: ", env_name, "| acc: ", env_acc * 1.0 / env_step)
        total_acc += env_acc
        total_step += env_step
    print("total acc: ", total_acc * 1.0 / total_step)