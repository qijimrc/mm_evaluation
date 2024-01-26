import torch
from PIL import Image

from transformers import AutoModelForCausalLM, LlamaTokenizer

import sys
# sys.path.insert(0, '/share/home/chengyean/evaluation/mm_evaluation/')
from mmbench.models.utils import CustomPromptModel, DATASET_TYPE
from mmbench.models.utils.misc import osp, timer

import string
import pandas as pd


class CogVLM:
    
    CACHE_DIR = '/share/official_pretrains/hf_home'
    @timer('init')
    def __init__(self, 
                 name: str='cogvlm-chat-hf', 
                 tokenizer_name: str='vicuna-7b-v1.5', 
                #  tokenizer_name: str='lmsys/vicuna-7b-v1.5', 
                 **kwargs) -> None:
        tokenizer_path = osp.join(cls.CACHE_DIR, tokenizer_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, 
                                                        local_files_only=True)
        vlm_path = osp.join(cls.CACHE_DIR, name)
        from accelerate import init_empty_weights
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{vlm_path}",
                # f"THUDM/{name}-hf",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cls.CACHE_DIR,
                local_files_only=True,
            ).to('cuda').eval()
        print(f"Loaded model from {vlm_path}")
    
    @timer('generate')
    def generate(self, image_path, prompt, dataset=None):

        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # print(tokenizer.decode(outputs[0]))
            response = self.tokenizer.decode(outputs[0])
        # output = response[len(prompt):]
        return response
    
if __name__ == '__main__':
    from pdb import set_trace as st
    model = CogVLM()
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    st()
    print(resp)

    
        
    