clip_env_prompts = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
    "drawer-open-v2": "The drawer is opened.", # let's try the flipped version.
    "sweep-into-v2": "The green cube is in the hole.", # unsolved there is reward issue
    "soccer-v2": "The soccer ball is in the goal.", # not solved, there is reward issue
}

task_prompt = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
    'drawer-open-v2': "to open the drawer",
    'sweep-into-v2': "to minimize the distance between the green cube and the hole",
    'soccer-v2': "to move the soccer ball into the goal"
}

image_0_prompt =  """
Consider the following two images:
Image 1:
"""
image_1_prompt = """
Image 2:
"""

# one stage
query_prompt = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
Reply a single line of 1 if yes, otherwise 0.
"""

#one stage + chain of thought
query_CoT_prompt = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
4. Is Image 2 more likely to achieve the goal? 1 if yes, otherwise 0.
"""

# two stage + chain of thought
thought_prompt = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

summary_prompt = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is Image 2 more likely to achieve the goal? 
Reply a single line of 1 if yes, otherwise 0.
"""