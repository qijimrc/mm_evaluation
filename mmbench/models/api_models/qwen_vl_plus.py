import logging
import copy as cp

import dashscope
from dashscope import MultiModalConversation
from mmdoctor.models.utils import osp, timer
from mmdoctor.common.registry import Registry
from mmdoctor.common.logger import log

from .base_api import BaseAPI

@Registry.register_model('QwenVLPlus')
@Registry.register_model('QwenVLMax')
class QwenVL(BaseAPI):

    is_api: bool = True

    def __init__(self, 
                 cfg_params,
                 args):
        super().__init__()
        assert cfg_params.api_key is not None, \
            "Please set the API Key (obtain it here: https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)"

        self.max_tokens = 1024
        self.temperature = args.temperature
        self.model = cfg_params.model_path
        self.system_prompt = cfg_params.system_prompt
        dashscope.api_key = cfg_params.api_key
    
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
                
    @timer('generate')
    def generate_inner(self, image_path, prompt, history=[]):
        inputs = [image_path, prompt]
        # model = 'qwen-vl-plus' 
        messages = self.build_msgs(msgs_raw=inputs, system_prompt=self.system_prompt)
        # gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)    
        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if response.status_code != 200:
                log(f'Error code: {response.status_code}, {response.message}', level=logging.ERROR)
                return -1, ''
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer
        except Exception as err:
            log(err, level=logging.ERROR)
            return -1, ''
