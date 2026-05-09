import torch

class Gemma:
    def __init__(self, version):
        from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        if version == "3-1b":
            model_id = "google/gemma-3-1b-it"
        elif version == "3-4b":
            model_id = "google/gemma-3-4b-it"
        elif version == "3-12b":
            model_id = "google/gemma-3-12b-it"
        elif version == "3-27b":
            model_id = "google/gemma-3-27b-it"
        if version == "3-1b":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = Gemma3ForCausalLM.from_pretrained(model_id, device_map="cuda", quantization_config=quantization_config).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
            self.tokenizer = AutoProcessor.from_pretrained(model_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def query_1(self, query_list, verbose=False):
        image1, image2, query_prompt = query_list
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."},]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is Image 1:"},
                        {"type": "image", "image": image1},
                        {"type": "text", "text": "This is Image 2:"},
                        {"type": "image", "image": image2},
                        {"type": "text", "text": query_prompt},
                    ]
                },
            ],
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)
        except:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device).to(torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)[0][input_len:]

        outputs = self.tokenizer.decode(outputs, skip_special_tokens=True)
        if(verbose):
            print(outputs)
        return outputs

