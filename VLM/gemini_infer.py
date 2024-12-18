import pathlib
import textwrap
import os
from PIL import Image
import google.generativeai as genai
import time
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

model = genai.GenerativeModel('gemini-1.5-flash')
text_model = genai.GenerativeModel('gemini-1.5-flash')
        
def gemini_query_1(query_list, temperature=0):
    beg = time.time()

    success = False
    try_cnt = 0
    while not success:
        try:
            response = model.generate_content(query_list,
            # response = model.generate_content([prompt, image1, image2],
                                            generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    # candidate_count=1,
                    # stop_sequences=['x'],
                    # max_output_tokens=20,
                    temperature=temperature),
                safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )
            print(response.text)
            response.resolve()
            success = True    
        except:
            print("gemini retrying...")
            time.sleep(3)
            try_cnt += 1
            # if try_cnt >= 5:
            #     break

    end = time.time()
    print("time elapsed: ", end - beg)
    if success:
        try:
            return response.text.split("\n")[0].strip().lstrip()
        except:
            return -1
    else:
        return -1

def gemini_query_2(query_list, summary_prompt, temperature=0):
    beg = time.time()

    success = False
    try_cnt = 0
    while not success:
        try:
            response = model.generate_content(query_list,
                                            generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    # candidate_count=1,
                    # stop_sequences=['x'],
                    # max_output_tokens=20,
                    temperature=temperature),
                safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )

            print(response.text)
            response.resolve()
    
            summary_response = text_model.generate_content(
                    summary_prompt.format(response.text),
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                    ),
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )
            print(summary_response.text)
            summary_response.resolve()
            success = True    
        except:
            # print("gemini retrying...")
            time.sleep(2)
            try_cnt += 1
            # if try_cnt >= 5:
            #     break

    

    end = time.time()
    if success:
        print("time elapsed: ", end - beg)
        try:
            return summary_response.text.split("\n")[0].strip().lstrip()
        except:
            return -1
    else:
        return -1

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    from prompt import image_0_prompt, image_1_prompt, query_prompt, thought_prompt, summary_prompt, query_CoT_prompt, task_prompt
    image_dir = "./data/"

    table_acc = {}
    table_step = {}
    if not os.path.exists(f"./result/gemini"):
        os.makedirs(f"./result/gemini")

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
                    ans = gemini_query_1(
                        [
                            image_0_prompt,
                            image_0, 
                            image_1_prompt,
                            image_1, 
                            query_prompt.format(task_prompt[env_name]),
                        ],
                        )

                    # ans = gemini_query_1(
                    #     [
                    #         image_0_prompt,
                    #         image_0, 
                    #         image_1_prompt,
                    #         image_1, 
                    #         query_CoT_prompt.format(task_prompt[env_name]),
                    #     ],
                    #     )

                    # ans = gemini_query_2(
                    #     [
                    #         image_0_prompt,
                    #         image_0, 
                    #         image_1_prompt,
                    #         image_1, 
                    #         thought_prompt.format("hunt a cow"),
                    #     ],
                    #     summary_prompt.format("hunt a cow", "{}"),
                    #     )

                    print("respon ans: ", ans)
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
    with open(f"./result/gemini/accuracy.txt", "a") as f:
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