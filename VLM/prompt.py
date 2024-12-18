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

task_prompt = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
}