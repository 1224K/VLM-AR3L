# transformers==4.38.2
class deepseekVL:
    def __init__(self, version="2_tiny"):
        import torch
        from transformers import AutoModelForCausalLM
        from .deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        # specify the path to the model
        if version == "2_tiny":
            model_path = "deepseek-ai/deepseek-vl2-tiny"
        elif version == "2_small":
            model_path = "deepseek-ai/deepseek-vl2-small"
        elif version =="2":
            model_path = "deepseek-ai/deepseek-vl2"

        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    def query_1(self, query_list):
        image1, image2, query_prompt = query_list
        conversation = [
            {
                "role": "<|User|>",
                "content": "This is image_1: <image>\n"
                            "This is image_2: <image>\n"
                            f'{query_prompt}',
                "images": [
                    image1,
                    image2,
                ],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[image1, image2],
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)


        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        attention_mask = prepare_inputs.attention_mask

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer
    
    def query_2(self, query_list, verbose=False):
        image1, image2, query_prompt, summary_prompt = query_list
        conversation = [
            {
                "role": "<|User|>",
                "content": "This is image_1: <image>\n"
                            "This is image_2: <image>\n"
                            f'{query_prompt}',
                "images": [
                    image1,
                    image2,
                ],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[image1, image2],
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)


        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        attention_mask = prepare_inputs.attention_mask

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        res1 = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        if verbose:
            print(f"{prepare_inputs['sft_format'][0]}", res1)

        
        return answer
    