
import torch
from PIL import Image
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

from mmdoctor.models.base_model import BaseModel
from mmdoctor.common.registry import Registry
from mmdoctor.models.utils import osp, get_rank_and_world_size, timer
@Registry.register_model('Emu2')
@Registry.register_model('Emu2_Chat')
class Emu2Wrapper(BaseModel):
    
    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs) -> None:
        """Init Function

        Args:
            args (_type_): _description_

        Raises:
            ValueError: _description_
        """
        
        model_name = cfg.model_path
        
        snapshot_name = os.listdir(osp.join(args.model_cache_dir, 
                                            model_name, 
                                            'snapshots'))[0]
        
        self.snapshot_path = osp.join(args.model_cache_dir, 
                                      model_name, 
                                      'snapshots', 
                                      snapshot_name)
        
        tokenizer = AutoTokenizer.from_pretrained(self.snapshot_path, 
                                        local_files_only=True,) 
        self.tokenizer = tokenizer
        
        num_devices = cfg.num_devices
        if num_devices == 1:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    self.snapshot_path,
                    device_map='cpu',
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True, 
                    local_files_only=True,).cuda().eval()
        elif num_devices == 2:
            local_rank,world_size = get_rank_and_world_size()
        
            device_num = torch.cuda.device_count()
            assert world_size * 2 <= device_num, \
            'The number of devices does not match the world size'
        
            device_1 = local_rank
            device_2 = local_rank + world_size
            torch.cuda.set_device(device_1)
            torch.cuda.set_device(device_2)
        
            tokenizer = AutoTokenizer.from_pretrained(self.snapshot_path, 
                                            local_files_only=True,) # "BAAI/Emu2-Chat"
            self.tokenizer = tokenizer
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    self.snapshot_path , # "BAAI/Emu2-Chat"
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
        else:
            raise ValueError(f'num_devices {num_devices} not supported')
        # input and output logits should be on same device

        self.model = model
        self.kwargs = dict(max_new_tokens= 64, length_penalty= -1)

    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.model.build_input_ids(
            text=[prompt],
            tokenizer=self.tokenizer,
            image=[image]
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                **self.kwargs)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text[0]


if __name__ == '__main__':
    from pdb import set_trace as st
    model = Emu2(name='emu2')
    image_path = '/share/home/chengyean/evaluation/data/dummy_example/image.png'
    prompt = 'Describe the image in details:'
    resp = model.generate(image_path=image_path, prompt=prompt,)
    print("*******")
    print(resp)

    