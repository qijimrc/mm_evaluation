import time
import random as rd

from mmdoctor.models.base_model import BaseModel
from mmdoctor.common.logger import log

class BaseAPI(BaseModel):
    is_api: bool = True
    fail_msg = "<RequestError></RequestError>"
    
    def __init__(self,
                 retry=3, 
                 wait=5):
        self.retry = retry
        self.wait = wait
        super().__init__()
        
    @NotImplementedError
    def generate_inner(self, image_path, prompt, history=[]):
        """get responses by api
        Args:
            image_path (str): image path
            prompt (str): user query
            history (list, optional): [(q1, a1), (q2, a2), ...]
        Return:
            ret_code (int): status code returned by api
                - 0: success
                - other: error
            answer (str): response
        """
        pass

    def generate(self, image_path, prompt, history=[]):
        answer = self.fail_msg
        for i in range(self.retry):
            T = rd.random() * self.wait * 2
            time.sleep(T)
            ret_code, answer = self.generate_inner(image_path, prompt, history)
            if ret_code == 0:
                return answer
            else:
                log(f"Request error, retry ({i}/{self.retry})")
        return answer