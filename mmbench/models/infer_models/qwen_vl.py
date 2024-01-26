import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
# from vlmeval.smp import isimg

# the following call should be changed to a 
import sys #
# sys.path.insert(0, '/share/home/chengyean/evaluation/mm_evaluation/')
from mmbench.models.utils.misc import timer, osp, isimg


class QwenVL:

    INSTALL_REQ = False
    # CACHE_DIR = '/share/official_pretrains/hf_home'
    CACHE_DIR = '/share/home/chengyean/evaluation/cya_ws/'

    def __init__(self, model_path='qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = osp.join(cls.CACHE_DIR, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                       trust_remote_code=True,     
                                                       local_files_only=True,)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                          device_map='cuda', 
                                                          trust_remote_code=True, 
                                                          local_files_only=True,).eval()
        # from pdb import set_trace; set_trace() 
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        vl_list = [{'image': img} for img in image_paths] + [{'text': prompt}]
        query = self.tokenizer.from_list_format(vl_list)
        
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    # def interleave_generate(self, ti_list, dataset=None):
    #     vl_list = [{'image': s} if isimg(s) else {'text': s} for s in ti_list]
    #     query = self.tokenizer.from_list_format(vl_list)
        
    #     inputs = self.tokenizer(query, return_tensors='pt')
    #     inputs = inputs.to(self.model.device)
    #     pred = self.model.generate(**inputs, **self.kwargs)
    #     response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
    #     response = response.split(query)[1].split('<|endoftext|>')[0]
    #     return response
    
class QwenVLChat:

    INSTALL_REQ = False
    CACHE_DIR = '/share/home/chengyean/evaluation/cya_ws/'
    @timer('init')
    def __init__(self, model_path='qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        
        self.model_path = osp.join(cls.CACHE_DIR, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                       trust_remote_code=True,     
                                                       local_files_only=True,)
        from accelerate import init_empty_weights
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                            device_map='cuda', 
                                                            trust_remote_code=True, 
                                                            local_files_only=True,).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        
    @timer('generate')
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        
        return response
    
    # def multi_generate(self, image_paths, prompt, dataset=None):
    #     vl_list = [{'image': img} for img in image_paths] + [{'text': prompt}]
    #     query = self.tokenizer.from_list_format(vl_list)    

    #     response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
    #     return response
    

if __name__ == '__main__':
    from pdb import set_trace as st
    # model = QwenVL()
    model = QwenVLChat()
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    # st()
    print(resp)
    