from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import time
import os
model_id = "microsoft/Phi-3.5-vision-instruct" 

class phi:
    def __init__(self):
        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
        )
        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=4
        )

    def query_1(self, query_list): 
        # beg = time.time()
        images = []
        placeholder = ""
        for i in range(2): 
            images.append(query_list[i])
            placeholder += f"<|image_{i+1}|>\n"

        messages = [
            {"role": "user", "content": placeholder+query_list[2]},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = { 
            "max_new_tokens": 1000, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0] 

        # print(response)
        # end = time.time()
        # print("time elapsed: ", end - beg)
        return response[-1].strip().lstrip()

    def query_2(self, query_list, summary_prompt): 
        beg = time.time()
        images = []
        placeholder = ""
        for i in range(2): 
            images.append(query_list[i])
            placeholder += f"<|image_{i+1}|>\n"

        messages = [
            {"role": "user", "content": placeholder+query_list[2]},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0] 

        print(response)
        
        # summary
        messages = [
            {"role": "user", "content": placeholder+summary_prompt.format(response)},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generate_ids = self.model.generate(**inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        
        print(response)
        end = time.time()
        print("time elapsed: ", end - beg)
        return response[-1].strip().lstrip()

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    from prompt import query_prompt, query_CoT_prompt, thought_prompt, summary_prompt, task_prompt

    vlm = phi()
    image_dir = "./data/"

    table_acc = {}
    table_step = {}
    # mkdir result
    if not os.path.exists(f"./result/phi"):
        os.makedirs(f"./result/phi")

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
                    ans = vlm.query_1(
                        [
                            image_0,
                            image_1,
                            query_prompt.format(task_prompt[env_name]),
                        ],
                        )
                    # ans = phi_query_1(
                    #     [
                    #         image_0,
                    #         image_1,
                    #         query_CoT_prompt.format("hunt a cow"),
                    #     ],
                    # )
                    # ans = phi_query_2(
                    #     [
                    #         image_0,
                    #         image_1,
                    #         thought_prompt.format("hunt a cow"),
                    #     ],
                    #     summary_prompt.format("hunt a cow", "{}"),
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
    with open(f"./result/phi/accuracy.txt", "a") as f:
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