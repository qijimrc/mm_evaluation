import os
import sys
# sys.path.insert(0, '/share/home/chengyean/evaluation/mm_evaluation/')
from mmbench.models.api_models.base import BaseAPI

import dashscope
from dashscope import MultiModalConversation
from mmbench.models.utils import osp, timer
from mmbench.models.base_model import BaseModel
from mmbench.common.registry import Registry

import copy as cp
import random as rd
import time

class QwenVLAPI(BaseAPI):

    def __init__(self, 
                 retry: int = 5,
                 wait: int = 5, 
                 key: str = None,
                 verbose: bool = False, 
                 temperature: float = 0.0, 
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 model: str = 'qwen-vl-plus',
                 **kwargs):

        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, "Please set the API Key (obtain it here: https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)"
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
    
    @staticmethod
    def build_msgs(msgs_raw, system_prompt=None):
        msgs = cp.deepcopy(msgs_raw) 
        ret = []
        if system_prompt is not None:
            content = list(dict(text=system_prompt))
            ret.append(dict(role='system', content=content))
        content = []
        for i, msg in enumerate(msgs):
            if osp.exists(msg):
                content.append(dict(image='file://' + msg))
            elif msg.startswith('http'):
                content.append(dict(image=msg))
            else:
                content.append(dict(text=msg))
        ret.append(dict(role='user', content=content))
        return ret
                
    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = True
        if isinstance(inputs, list):
            for pth in inputs:
                if osp.exists(pth) or pth.startswith('http'):
                    pure_text = False
        assert not pure_text
        # model = 'qwen-vl-plus' 
        messages = self.build_msgs(msgs_raw=inputs, system_prompt=self.system_prompt)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)    
        gen_config.update(self.kwargs)
        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if self.verbose:
                print(response)            
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f"The input messages are {inputs}.")

            return -1, '', ''

    def generate(self, inputs, **kwargs):
        input_type = None
        if isinstance(inputs, str):
            input_type = 'str'
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            input_type = 'strlist'
        elif isinstance(inputs, list) and isinstance(inputs[0], dict):
            input_type = 'dictlist'
        assert input_type is not None, input_type

        answer = None
        for i in range(self.retry):
            T = rd.random() * self.wait * 2
            time.sleep(T)
            try:
                ret_code, answer, log = self.generate_inner(inputs, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    self.logger.info(f"RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}")
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}:')
                    self.logger.error(err)
        
        return self.fail_msg if answer in ['', None] else answer
        
        

@Registry.register_model('QwenVLPlus')
@Registry.register_model('QwenVLMax')
class QwenVLWrapper(QwenVLAPI, BaseModel):
    
    is_api: bool = True
    
    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs) -> None:
        model_name = cfg.model_name
        super(QwenVLWrapper, self).__init__(model_name=model_name)
    
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        return super(QwenVLAPI, self).generate([image_path, prompt])        


if __name__ == '__main__':
    from pdb import set_trace as st
    model = QwenVLWrapper(model_name='qwen-vl-plus')
    model = QwenVLWrapper(model_name='qwen-vl-max')
    
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    print(resp)
