class gpt:
    def __init__(self):
        from openai import OpenAI
        self.model = "gpt-4.1-nano"
        self.client = OpenAI()
        self.total_tokens = 0

    # Function to encode the image
    def encode_image(self, image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def query_1(self, image_1_path, image_2_path, env_query, temperature=0, verbose=False):
        image_1 = self.encode_image(image_1_path)
        image_2 = self.encode_image(image_2_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {   "type": "text", "text": "This is image 1: " },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_1}",
                            },
                        },
                        {   "type": "text", "text": "This is image 2: " },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_2}",
                            },
                        },
                        {   "type": "text", "text": env_query },
                    ],
                }
            ],
            temperature=temperature,
        )
        
        if verbose:
            print(response.choices[0].message.content)
        return response.choices[0].message.content

    def query_2(self, image_1_path, image_2_path, env_thought, env_summary, temperature=0, verbose=False):
        image_1 = self.encode_image(image_1_path)
        image_2 = self.encode_image(image_2_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {   "type": "text", "text": "This is image 1: " },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_1}",
                            },
                        },
                        {   "type": "text", "text": "This is image 2: " },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_2}",
                            },
                        },
                        {   "type": "text", "text": env_thought },
                    ],
                }
            ],
            temperature=temperature,
        )
        
        if verbose:
            print(response.choices[0].message.content)
            self.total_tokens += response.usage.completion_tokens
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {   "type": "text", "text": env_summary.format(response.choices[0].message.content) },
                    ],
                }
            ],
            temperature=temperature,
        )
        if verbose:
            print("Summary:")
            print(response.choices[0].message.content)
            self.total_tokens += response.usage.completion_tokens

        return response.choices[0].message.content