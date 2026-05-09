# torch >= 2.3.1
class QwenVL:
    def __init__(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        # specify the path to the model
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def query_1(self, query_list):
        from qwen_vl_utils import process_vision_info
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

        text_stage1 = self.processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
        image_inputs_stage1, _ = process_vision_info(messages_stage1)
        inputs_stage1 = self.processor(
            text=[text_stage1],
            images=image_inputs_stage1,
            padding=True,
            return_tensors="pt",
        )
        inputs_stage1 = inputs_stage1.to("cuda")

        generated_ids_stage1 = self.model.generate(**inputs_stage1, max_new_tokens=500)
        generated_ids_trimmed_stage1 = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage1.input_ids, generated_ids_stage1)
        ]
        result_stage1 = self.processor.batch_decode(
            generated_ids_trimmed_stage1, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # print("First stage result:", result_stage1)

        return result_stage1.split("\n")[-1].strip().lstrip()

    def query_2(self, query_list, summary_prompt):
        from qwen_vl_utils import process_vision_info
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

        text_stage1 = self.processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
        image_inputs_stage1, _ = process_vision_info(messages_stage1)
        inputs_stage1 = self.processor(
            text=[text_stage1],
            images=image_inputs_stage1,
            padding=True,
            return_tensors="pt",
        )
        inputs_stage1 = inputs_stage1.to("cuda")

        generated_ids_stage1 = self.model.generate(**inputs_stage1, max_new_tokens=500)
        generated_ids_trimmed_stage1 = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage1.input_ids, generated_ids_stage1)
        ]
        result_stage1 = self.processor.batch_decode(
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

        text_stage2 = self.processor.apply_chat_template(messages_stage2, tokenize=False, add_generation_prompt=True)
        inputs_stage2 = self.processor(
            text=[text_stage2],
            padding=True,
            return_tensors="pt",
        )
        inputs_stage2 = inputs_stage2.to("cuda")
        # inputs_stage2 = inputs_stage2.to("cpu")

        generated_ids_stage2 = self.model.generate(**inputs_stage2, max_new_tokens=500)
        generated_ids_trimmed_stage2 = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_stage2.input_ids, generated_ids_stage2)
        ]
        result_stage2 = self.processor.batch_decode(
            generated_ids_trimmed_stage2, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # print("Second stage result:", result_stage2)

        return result_stage2.split("\n")[-1].strip().lstrip()