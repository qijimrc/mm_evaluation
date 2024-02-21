import base64
import logging

from .base_api import BaseAPI
from mmdoctor.common.registry import Registry
from mmdoctor.models.utils import timer
from mmdoctor.common.logger import log

@Registry.register_model("GPT4V")
class GPT4V(BaseAPI):

    def __init__(self, cfg_params, args):
        super().__init__()
        self.model_version = cfg_params.model_version

        from openai import OpenAI
        self.client = OpenAI(api_key=cfg_params.api_key, base_url=cfg_params.url)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @timer('generate')
    def generate_inner(self, image_path, prompt, history=[]):
        payload = dict(
            model=self.model_version,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(image_path)}"
                            }
                        }
                    ]
                },
            ],
            max_tokens=1024
        )
        response = self.client.chat.completions.create(**payload)
        if response.status_code != 200:
            log(f'Error code: {response.status_code}, {response.message}', level=logging.ERROR)
            return -1, ''
        result = response.choices[0].message.content
        return 0, result
