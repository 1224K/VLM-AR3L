#--------------------------------- modify by K 
# 4.26.1
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer
import time
import torch
import os

# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

## MiniCPM-o 2.6
# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=False,
    init_tts=False
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

# In addition to vision-only mode, tts processor and vocos also needs to be initialized
# model.init_tts()

def MiniCPM_query(query_list, summary_prompt, temperature=0.1):
    beg = time.time()
    res = model.chat(
        image=None,
        msgs=[{'role': 'user', 'content': query_list}],
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        # temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)
    summary_prompt[-1] = summary_prompt[-1].format(res)
    summary_response = model.chat(
        image=None,
        msgs=[{'role': 'user', 'content': summary_prompt}],
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        # temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    print(summary_response)
    end = time.time()

    print("time elapsed: ", end - beg)

    return summary_response.split("\n")[-1].strip().lstrip()

def MiniCPM_query_single(query_list, temperature=0.1):
    beg = time.time()
    res = model.chat(
        image=None,
        msgs=[{'role': 'user', 'content': query_list}],
        tokenizer=tokenizer,
        sampling=False, # if sampling=False, beam_search will be used by default
        # temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)
    end = time.time()

    print("time elapsed: ", end - beg)

    return res.split("\n")[-1].strip().lstrip()

def MiniCPM_query_concatenate(query_list, summary_prompt, img, temperature=0.1):
    beg = time.time()
    res = model.chat(
        image=img,
        msgs=[{'role': 'user', 'content': query_list}],
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)

    summary_response = model.chat(
        image=img,
        msgs=[{'role': 'user', 'content': summary_prompt.format(res)}],
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    end = time.time()

    print("time elapsed: ", end - beg)

    return summary_response

def MiniCPM_query_single_concatenate(query_list, img, temperature=0.1):
    beg = time.time()
    res = model.chat(
        image=img,
        msgs=[{'role': 'user', 'content': query_list}],
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=temperature,
        # system_prompt='' # pass system_prompt if needed
    )
    end = time.time()

    print("time elapsed: ", end - beg)

    return res.split("\n")[-1].strip().lstrip()

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

def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img

def add_label(image, label_text, position=(10, 10), font_size=20, font_color=(255, 255, 255)):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default(font_size)  # Fallback to default font if arial not available
    draw.text(position, label_text, font=font, fill=font_color)
    return image


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    from prompt import image_0_prompt, image_1_prompt, query_prompt, thought_prompt, summary_prompt, query_CoT_prompt, task_prompt

    image_dir = "./data/"

    table_acc = {}
    table_step = {}
    # mkdir result
    if not os.path.exists(f"./result/MiniCPM"):
        os.makedirs(f"./result/MiniCPM")

    for env_name in os.listdir(image_dir):
        if os.path.isdir(image_dir + env_name) == False:
            continue
        if env_name in ["soccer-v2", "sweep-into-v2", "drawer-open-v2"]:
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

                # add label
                image_0 = add_label(image_0, "Image 1")
                image_1 = add_label(image_1, "Image 2")

                for _ in range(1):
                    ans = MiniCPM_query_single(
                        [
                            image_0_prompt,
                            image_0,
                            image_1_prompt,
                            image_1,
                            query_prompt.format(task_prompt[env_name]),
                        ],
                        )
                    # ans = MiniCPM_query_single(
                    #     [
                    #         image_0_prompt,
                    #         image_0,
                    #         image_1_prompt,
                    #         image_1,
                    #         query_CoT_prompt.format("hunt a cow"),
                    #     ],
                    # )
                    # ans = MiniCPM_query(
                    #     [
                    #         image_0_prompt,
                    #         image_0,
                    #         image_1_prompt,
                    #         image_1,
                    #         thought_prompt.format("hunt a cow"),
                    #     ],
                    #     [
                    #         image_0_prompt,
                    #         image_0,
                    #         image_1_prompt,
                    #         image_1,
                    #         summary_prompt.format("hunt a cow", "{}"),
                    #     ],
                    #     )


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
    with open(f"./result/MiniCPM/accuracy.txt", "a") as f:
        f.write(f"------------------------------------------------------------------------\n")
        for env_name in table_acc:
            f.write(f"prompt: {query_prompt.format(task_prompt[env_name])}\n")
            env_acc = 0
            env_step = 0
            for label in table_acc[env_name]:
                env_acc += table_acc[env_name][label]
                env_step += table_step[env_name][label]
                print("env_name: ", env_name, "| label: ", label, "| acc: ", table_acc[env_name][label] * 1.0 / table_step[env_name][label], "| ac ", table_acc[env_name][label] , "| step: ", table_step[env_name][label])
                f.write(f"env_name: {env_name} | label: {label} | acc: {table_acc[env_name][label] * 1.0 / table_step[env_name][label]} | ac {table_acc[env_name][label]} | step: {table_step[env_name][label]}\n")
            print("env_name: ", env_name, "| acc: ", env_acc * 1.0 / env_step)
            f.write(f"env_name: {env_name} | acc: {env_acc * 1.0 / env_step}\n")
            total_acc += env_acc
            total_step += env_step
        print("total acc: ", total_acc * 1.0 / total_step)
        f.write(f"total acc: {total_acc * 1.0 / total_step}\n")