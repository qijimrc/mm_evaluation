import torch
import transformers
import argparse


class LlamaChatter():
    def __init__(self,
                 model_version='llama3-8b-instruct',
                 device =1,
                 ) -> None:
        self.device = device

        if model_version == 'llama3-8b-instruct':
            self.pipeline = transformers.pipeline(
                "text-generation",
                model='/share/official_pretrains/hf_home/Meta-Llama-3-8B-Instruct/',
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device,
            )
        else:
            raise BaseException

    def chat(self, prompt:str="Who are you?") -> str:
        messages = [
            {"role": "system", "content": "You are a an assistant who always responds user's question by following their strict requirements."},
            {"role": "user", "content": str(prompt)},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='How are you?')
    args = parser.parse_args()

    chatter = LlamaChatter()
    response = chatter.chat(args.prompt)
    print(response)


