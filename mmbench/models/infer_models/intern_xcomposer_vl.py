import torch
from transformers import AutoModel, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image


from accelerate import init_empty_weights

from mmbench.models.utils import osp, timer
from mmbench.common.registry import Registry
from mmbench.models.base_model import BaseModel

@Registry.register_model('XComposer')
class XComposer(BaseModel):
    
    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs):
        
        model_path = cfg.model_name
        assert model_path is not None
        self.model_path = osp.join(args.model_cache_dir, model_path)
        try:
            import rotary_emb
        except:
            raise ValueError("Please install rotary-emb")
            # install guide:
            # git clone git@github.com:Dao-AILab/flash-attention.git
            # cd flash-attention
            # cd csrc 
            # rm -r cutlass
            # git clone git@github.com:NVIDIA/cutlass.git
            # python setup.py install
            # cd csrc/rotary
            # pip install -e .
            
        with init_empty_weights():
            model = AutoModel.from_pretrained(self.model_path,
                                            device_map='cpu', 
                                            local_files_only=True, 
                                            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                  trust_remote_code=True, 
                                                  local_files_only=True,)
        model.tokenizer = tokenizer
        self.model = model
        self.device = self.model.internlm_model.model.embed_tokens.weight.device
        stop_words_ids = [
            torch.tensor([103027]).to(self.device), ### end of human
            torch.tensor([103028]).to(self.device), ### end of bot
        ]
        default_kwargs = {
            'max_new_tokens': 128, 'num_beams': 5, 'do_sample': False, 
            'min_length': 1, 'repetition_penalty': 1.5, 'length_penalty': 1.0
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
    
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        return self.model.generate(prompt, image_path, **self.kwargs)
    
if __name__ == '__main__':
    from pdb import set_trace as st
    model = XComposer()
    image_path = '/share/home/chengyean/evaluation/data/dummy_example/image.png'
    prompt = 'Describe the image in details:'
    resp = model.generate(image_path=image_path, prompt=prompt,)
    print("*******")
    print(resp)

    