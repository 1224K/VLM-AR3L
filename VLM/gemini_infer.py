class Gemini:
    def __init__(self, version="2.0"):
        import os
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        if version == "2.5":
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        elif version == "2.0":
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        elif version == "1.5":
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens = 0
    
    def query_1(self, images, query, temperature=0, verbose=False):
        import google.generativeai as genai
        import time

        
        query_list = []
        for (i, image) in enumerate(images):
            query_list.append(f"This is Image {i+1}: ")
            query_list.append(image)
        query_list.append(query)
        print("query_list:", query_list)
        success = False
        try_cnt = 0
        while not success:
            try:
                response = self.model.generate_content(query_list,
                        generation_config=genai.types.GenerationConfig(
                        temperature=temperature),
                )
                if verbose:
                    print(response.text)
                response.resolve()
                success = True    
            except:
                print("gemini retrying...")
                time.sleep(3)
                try_cnt += 1
                # if try_cnt >= 5:
                #     break

        if success:
            return response.text
        else:
            return -1

    def query_token(self, query_list, temperature=0, verbose=False):
        import google.generativeai as genai
        import time

        image_1, image_2, env_query = query_list
        query_list = ["This is image 1: ", image_1, "This is image 2: ", image_2, env_query]
        success = False
        try_cnt = 0
        while not success:
            try:
                tokens = self.model.count_tokens(
                    contents=query_list,
                ).total_tokens
                print("Input tokens:", tokens)
                response = self.model.generate_content(query_list,
                        generation_config=genai.types.GenerationConfig(
                        temperature=temperature),                    
                )
                print(response.text)
                print(response.usage_metadata.candidates_token_count)
                response.resolve()
                success = True    
            except Exception as e:
                print("Error:", e)
                print("gemini retrying...")
                time.sleep(3)
                try_cnt += 1
                # if try_cnt >= 5:
                #     break

        if success:
            return response.text
        else:
            return -1

    def query_2(self, query_list, summary_prompt, temperature=0, verbose=False):
        import google.generativeai as genai
        import time

        image_1, image_2, env_query = query_list
        query_list = ["This is image 1: ", image_1, "This is image 2: ", image_2, env_query]
        success = False
        try_cnt = 0
        while not success:
            try:
                response = self.model.generate_content(query_list,
                                                generation_config=genai.types.GenerationConfig(
                        temperature=temperature),
                )
                if verbose:
                    print(response.text)
                    self.total_tokens += response.usage_metadata.candidates_token_count
                response.resolve()
        
                summary_response = self.model.generate_content(
                        summary_prompt.format(response.text),
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                        ),
                )
                if verbose:
                    print("Summary response:")
                    print(summary_response.text)
                    self.total_tokens += summary_response.usage_metadata.candidates_token_count
                summary_response.resolve()
                success = True    
            except Exception as e:
                print("Error:", e)
                print("gemini retrying...")
                time.sleep(2)
                try_cnt += 1
                # if try_cnt >= 5:
                #     break

        if success:
            try:
                return summary_response.text.split("\n")[0].strip().lstrip()
            except:
                return -1
        else:
            return -1