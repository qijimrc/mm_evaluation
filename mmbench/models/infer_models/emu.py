
import torch
from PIL import Image

import sys
# sys.path.insert(0, '/share/home/chengyean/evaluation/mm_evaluation/')
# from mmbench.models.utils import CustomPromptModel, DATASET_TYPE
from mmbench.models.utils.misc import osp, splitlen, get_rank_and_world_size, \
    isimg, timer

import warnings
class Emu:

    CACHE_DIR = '/share/home/chengyean/evaluation/cya_ws/hf_down'
    @timer('init')
    def __init__(self, 
                 name, 
                 model_path_map={
                     "emu2": "BAAI/Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a",
                     "emu2_chat": "AI-ModelScope/Emu2-Chat"
                     }, 
                 **kwargs):
        
        if name in model_path_map:
            model_path = model_path_map[name]
        else:
            raise ValueError(f"Model name {name} not found in model_path_map "
                             f"{model_path_map}")
        
        self.model_path = osp.join(cls.CACHE_DIR, model_path)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
        
        local_rank,world_size = get_rank_and_world_size()
        
        device_num = torch.cuda.device_count()
        assert world_size * 2 <= device_num, \
        'The number of devices does not match the world size'
        
        device_1 = local_rank
        device_2 = local_rank + world_size
        torch.cuda.set_device(device_1)
        torch.cuda.set_device(device_2)
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path , 
                                        local_files_only=True,) # "BAAI/Emu2-Chat"
        self.tokenizer = tokenizer
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path , # "BAAI/Emu2-Chat"
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True, 
                local_files_only=True,)  

        device_map = infer_auto_device_map(model, 
                                           max_memory={device_1:'38GiB', 
                                                       device_2:'38GiB',}, 
                                           no_split_module_classes=['Block',
                                                                    'LlamaDecoderLayer'])  
        # input and output logits should be on same device
        device_map["model.decoder.lm.lm_head"] = device_1
        
        model = dispatch_model(
            model, 
            device_map=device_map).eval()
        
        self.model = model
        
        kwargs_default = dict(max_new_tokens= 64, length_penalty= -1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
    
    def interleave_generate(self, ti_list, dataset=None):
        query, images = '',[]
        for item in ti_list:
            if isimg(item):
                images.append(Image.open(item).convert('RGB'))
                query += '[<IMG_PLH>]'
            else:
                query += item
        
        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=images
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                **self.kwargs)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text[0]
    
    @timer('generate')
    def generate(self, image_path, prompt, dataset=None):
        tl_list = [image_path,prompt]
        output = self.interleave_generate(tl_list, dataset)
        return output


if __name__ == '__main__':
    from pdb import set_trace as st
    model = Emu(name='emu2')
    image_path = '/share/home/chengyean/evaluation/data/dummy_example/image.png'
    prompt = 'Describe the image in details:'
    resp = model.generate(image_path=image_path, prompt=prompt,)
    print("*******")
    print(resp)

    