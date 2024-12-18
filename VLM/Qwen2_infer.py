from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import time
import torch
import os
import sys

# Initialize Qwen/Qwen2-VL-7B-Instruct model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

#-------------------------------------------------------------------------------------
# Initialize Qwen/Qwen2-VL-72B-Instruct model and processor

# offload_folder = "offload_Qwen2-VL-72B-Instruct"

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-72B-Instruct", 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto", 
#     offload_folder="./offload_folder"  # Specify the offload folder here
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

def Qwen_query_1(query_list, temperature=0.1):
    beg = time.time()

    messages_stage1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_list[0]},  # 第一張圖片
                {"type": "image", "image": query_list[1]},  # 第二張圖片
                {"type": "text", "text": query_list[2]},  # 文字描述
            ],
        }
    ]

    text_stage1 = processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
    image_inputs_stage1, _ = process_vision_info(messages_stage1)
    inputs_stage1 = processor(
        text=[text_stage1],
        images=image_inputs_stage1,
        padding=True,
        return_tensors="pt",
    )
    inputs_stage1 = inputs_stage1.to("cuda")

    generated_ids_stage1 = model.generate(**inputs_stage1, max_new_tokens=500)
    generated_ids_trimmed_stage1 = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage1.input_ids, generated_ids_stage1)
    ]
    result_stage1 = processor.batch_decode(
        generated_ids_trimmed_stage1, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("First stage result:", result_stage1)

    end = time.time()
    print("time elapsed: ", end - beg)

    return result_stage1.split("\n")[-1].strip().lstrip()

def Qwen_query_2(query_list, summary_prompt, temperature=0.1):
    beg = time.time()

    messages_stage1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_list[0]},  # 第一張圖片
                {"type": "image", "image": query_list[1]},  # 第二張圖片
                {"type": "text", "text": query_list[2]},  # 文字描述
            ],
        }
    ]

    text_stage1 = processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
    image_inputs_stage1, _ = process_vision_info(messages_stage1)
    inputs_stage1 = processor(
        text=[text_stage1],
        images=image_inputs_stage1,
        padding=True,
        return_tensors="pt",
    )
    inputs_stage1 = inputs_stage1.to("cuda")
    # inputs_stage1 = inputs_stage1.to("cpu")

    generated_ids_stage1 = model.generate(**inputs_stage1, max_new_tokens=500)
    generated_ids_trimmed_stage1 = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage1.input_ids, generated_ids_stage1)
    ]
    result_stage1 = processor.batch_decode(
        generated_ids_trimmed_stage1, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("First stage result:", result_stage1)

    messages_stage2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": summary_prompt.format(result_stage1)},  # 基於第一階段結果生成的查詢
            ],
        }
    ]

    text_stage2 = processor.apply_chat_template(messages_stage2, tokenize=False, add_generation_prompt=True)
    inputs_stage2 = processor(
        text=[text_stage2],
        padding=True,
        return_tensors="pt",
    )
    inputs_stage2 = inputs_stage2.to("cuda")
    # inputs_stage2 = inputs_stage2.to("cpu")

    generated_ids_stage2 = model.generate(**inputs_stage2, max_new_tokens=500)
    generated_ids_trimmed_stage2 = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage2.input_ids, generated_ids_stage2)
    ]
    result_stage2 = processor.batch_decode(
        generated_ids_trimmed_stage2, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("Second stage result:", result_stage2)

    end = time.time()
    print("time elapsed: ", end - beg)

    return result_stage2.split("\n")[-1].strip().lstrip()

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
                    # ans = Qwen_query_1(
                    #     [
                    #         image_0_path,
                    #         image_1_path,
                    #         query_prompt.format("hunt a cow"),
                    #     ],
                    #     )
                    # ans = Qwen_query_1(
                    #     [
                    #         image_0_path,
                    #         image_1_path,
                    #         query_CoT_prompt.format("hunt a cow"),
                    #     ],
                    # )
                    ans = Qwen_query_2(
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