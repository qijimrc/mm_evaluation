from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import init_empty_weights

from mmbench.models.utils import osp, timer
from mmbench.models.base_model import BaseModel
from mmbench.common.registry import Registry

@Registry.register_model('XComposer2')
class XComposer2(BaseModel):
    
    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs):
        model_path = cfg.model_name
        assert model_path is not None
        self.model_path = osp.join(args.model_cache_dir, model_path)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                            device_map='cpu', 
                                            local_files_only=True, 
                                            trust_remote_code=True).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                revision='master',
                                                trust_remote_code=True, 
                                                local_files_only=True,)
        self.history = []
       
        self.model.tokenizer = self.tokenizer
        
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        response, history = self.model.chat(query=prompt, image=image_path, tokenizer=self.tokenizer,history=self.history)
        self.history = history
        return response
        

if __name__ == '__main__':
    from pdb import set_trace as st
    model = XComposer2()
    image_path = '/share/home/chengyean/evaluation/data/dummy_example/math-5.png'
    text = """如图,D是等边三角形ABC的边AC上一点,四边形CDEF是平行四边形,点F在BC的延长线上,G为BE的中点.连接DG,若AB=10,AD=DE=4,则DG的长为()"""
    response, history = model.chat(prompt=text, image_path=image_path, history=[])
    print(response)