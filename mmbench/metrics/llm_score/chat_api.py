import json
import urllib3
import requests

from sat.helpers import print_rank0

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ChatAPI():
    def __init__(self) -> None:
        self.api_config = {
            "chatglm2-66b": {
                "url": "https://117.161.233.26:8443/v1/completions",
                "headers": {
                    "Content-Type": "application/json",
                    "Host": "infra-research-8k.glm.ai"
                },
                "parameters": {
                    "model": "chatglm2",
                    "do_sample": False,
                    "max_tokens": 2048,
                    "stream": False,
                    "seed": 1234
                },
                "post_processor": self._post_chatglm2_66b,
                "pre_processor": self._pre_chatglm2_66b
            },
            "chatgpt": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer ***"
                },
                "parameters": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 1.0,
                    "max_tokens": 1024,
                    "top_p": 1.0
                },
                "post_processor": self._post_chatgpt,
                "pre_processor": self._pre_chatgpt
            }
        }
        self.failed = "Request Failed"

    def get_api_servers(self):
        return list(self.api_config.keys())
    
    def _post_chatglm2_66b(self, response):
        status, result = response.status_code, self.failed
        if status == 200:
            result = json.loads(response.text)["choices"][0]["text"].strip().replace("\n", "")
            status = "SUCCESS"
        return status, result

    def _post_chatgpt(self, response):
        return response

    def _pre_chatglm2_66b(self, prompt):
        update_parameters = {"prompt": prompt}
        return update_parameters
    
    def _pre_chatgpt(self, prompt):
        update_parameters = {"messages": [{"role": "user", "content": prompt}]}
        return update_parameters

    def get_response(self, api_server, prompt, **kwargs) -> str:
        config = self.api_config[api_server]
        parameters = config["parameters"]
        parameters.update(config["pre_processor"](prompt, **kwargs))
        status, result = -1, ""
        try:
            with requests.post(config["url"], 
                               headers=config["headers"],
                               json=parameters,
                               verify=False,
                               timeout=50) as response:
                status, result = config["post_processor"](response)
        except Exception as e:
            print_rank0(str(e))
        return status, result

if __name__ == "__main__":
    chatapi = ChatAPI()
    status, result = chatapi.get_response("chatglm2-66b", prompt="这张图片展示了什么\n")
    print(status, result)