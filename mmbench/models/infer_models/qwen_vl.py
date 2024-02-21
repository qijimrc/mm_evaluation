import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from mmdoctor.models.utils import timer, osp
from mmdoctor.models.base_model import BaseModel
from mmdoctor.common.registry import Registry

@Registry.register_model('QwenVL')
class QwenVL(BaseModel):

    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs) -> None:
        
        model_path = cfg.model_path
        assert model_path is not None
        self.model_path = osp.join(args.model_cache_dir, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                       trust_remote_code=True,     
                                                       local_files_only=True,)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                          device_map='cuda', 
                                                          trust_remote_code=True, 
                                                          local_files_only=True,).eval()
        # from pdb import set_trace; set_trace() 
        self.kwargs = kwargs
    
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
@Registry.register_model('QwenVLChat')
class QwenVLChat(QwenVL):
    def __init__(self, cfg, args,
                 **kwargs): 
        super().__init__(cfg, args, **kwargs)
    
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        
        return response

if __name__ == '__main__':
    from pdb import set_trace as st
    # model = QwenVL()
    model = QwenVLChat()
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    # st()
    print(resp)
    