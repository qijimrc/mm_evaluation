import os
import sys
# sys.path.insert(0, '/share/home/chengyean/evaluation/mm_evaluation/')
from mmbench.models.api_models.base import BaseAPI

import dashscope
from dashscope import MultiModalConversation

import copy as cp
import os.path as osp

class QwenVLWrapper(BaseAPI):

    is_api: bool = True
    def __init__(self, 
                 retry: int = 5,
                 wait: int = 5, 
                 key: str = None,
                 verbose: bool = True, 
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

class QwenVLPlus(QwenVLWrapper):

    def generate(self, image_path, prompt, dataset=None):
        return super(QwenVLPlus, self).generate([image_path, prompt])
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        return super(QwenVLPlus, self).generate(image_paths + [prompt])
    
    def interleave_generate(self, ti_list, dataset=None):
        return super(QwenVLPlus, self).generate(ti_list)
            

if __name__ == '__main__':
    from pdb import set_trace as st
    model = QwenVLPlus(model='qwen-vl-plus')
    model = QwenVLPlus(model='qwen-vl-max')
    
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    print(resp)
