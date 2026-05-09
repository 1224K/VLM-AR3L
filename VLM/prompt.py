env_clip_prompts = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
    'shear_sheep': "shear a sheep",
    'harvest_water': "harvest a water",
    "drawer-open-v2": "The drawer is opened.", # let's try the flipped version.
    "sweep-into-v2": "The green cube is in the hole.", # unsolved there is reward issue
    "soccer-v2": "The soccer ball is in the goal.", # not solved, there is reward issue

    "CartPole-v1": "pole vertically upright on top of the cart.",
    
    "softgym_RopeFlattenEasy": "The blue rope is straightened.",
    "softgym_PassWater": "The container, which holds water, is as close to the red circle as possible without causing too many water droplets to spill.",
    "softgym_ClothFoldDiagonal": "The cloth is folded diagonally from top left corner to bottom right corner.",
}

task_prompt = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    # 'milk_cow': "find and milk a cow",
    'hunt_cow': "hunt a cow",
    # 'hunt_cow': "hunt a cow. If a cow is hit, it will turn red",
    'shear_sheep': "shear a sheep",
    'harvest_water': "harvest a water",
    # 'drawer-open-v2': "to open the drawer",
    'drawer-open-v2': "to maximize drawer opening",
    'sweep-into-v2': "to minimize the distance between the green cube and the hole",
    # 'soccer-v2': "to move the soccer ball into the goal"
    'soccer-v2': "to minimize the distance between the soccer ball and the goal",
    'CartPole-v1': "to balance the brown pole on the black cart to be upright",

    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}

# two label
## one stage
two_label_query_prompt = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
Reply a single line of 1 if yes, otherwise 0.
"""

two_label_env_query_prompt = {}
for env_name, prompt in task_prompt.items():
    two_label_env_query_prompt[env_name] = two_label_query_prompt.format(prompt)

## two stage
two_label_thought_prompt = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
"""

two_label_summary_prompt = """
Based on the text below to the question:
The goal is {}. Is Image 2 more likely to achieve the goal?
{}

Reply a single line of 1 if yes, otherwise 0.
"""

two_label_env_thought_prompt = {}
two_label_env_summary_prompt = {}
for env_name, prompt in task_prompt.items():
    two_label_env_thought_prompt[env_name] = two_label_thought_prompt.format(prompt)
    two_label_env_summary_prompt[env_name] = two_label_summary_prompt.format(prompt, "{}")

# three label
## one stage
query_prompt = """
The goal is {}.
Is the goal better achieved in Image 1 or Image 2? 
Reply a single line of 1 if the goal is better achieved in Image 1, or 2 if it is better achieved in Image 2.
Reply 0 if unsure or there is no difference."""
# query_prompt = """
# The goal is {}.  
# Relative to Image 1, does Image 2 move closer to the goal?  
# Reply with a single word yes or no."""

env_query_prompt = {}
for env_name, prompt in task_prompt.items():
    env_query_prompt[env_name] = query_prompt.format(prompt)

## two stage
thought_prompt = """
The goal is {}. Is the goal better achieved in Image 1 or Image 2? 
"""

CoT_prompt = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

summary_prompt = """
Based on the text below to the question:
The goal is {}. Is the goal better achieved in Image 1 or Image 2? 
{}

Reply a single line of 1 if the goal is better achieved in Image 1, or 2 if it is better achieved in Image 2.
Reply 0 if the text is unsure or there is no difference.
"""

env_thought_prompt = {}
env_summary_prompt = {}
env_query_CoT_prompt = {}
for env_name, prompt in task_prompt.items():
    env_thought_prompt[env_name] = thought_prompt.format(prompt)
    env_summary_prompt[env_name] = summary_prompt.format(prompt, "{}")
    env_query_CoT_prompt[env_name] = CoT_prompt.format(prompt)

query_triple_prompt = """
The goal is {}.
Which image (Image 1, Image 2, or Image 3) best achieves the goal, and which image least achieves the goal?
Output exactly one line in the format <BEST>,<WORST> using only the image identifiers.  
If unsure for either position, output 0 in that position.  
Examples:  
- 2, 1  
"""

env_query_triple_prompt = {}
for env_name, prompt in task_prompt.items():
    env_query_triple_prompt[env_name] = query_triple_prompt.format(prompt)

# #one stage + chain of thought
# query_CoT_prompt = """
# 1. What is shown in Image 1?
# 2. What is shown in Image 2?
# 3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
# 4. Is Image 2 more likely to achieve the goal? 1 if yes, otherwise 0.
# """

# env_query_CoT_prompt = {}
# for env_name, prompt in task_prompt.items():
#     env_query_CoT_prompt[env_name] = query_CoT_prompt.format(prompt)

# # two stage + chain of thought
# thought_prompt = """
# 1. What is shown in Image 1?
# 2. What is shown in Image 2?
# 3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
# """

# summary_prompt = """
# Based on the text below to the questions:
# 1. What is shown in Image 1?
# 2. What is shown in Image 2?
# 3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
# {}

# Is Image 2 more likely to achieve the goal? 
# Reply a single line of 1 if yes, otherwise 0.
# """