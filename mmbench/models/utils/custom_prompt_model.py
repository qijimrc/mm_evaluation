# from ..smp import *
# from .dataset_config import img_root_map
from abc import abstractmethod

class CustomPromptModel:

    @abstractmethod
    def use_custom_prompt(self, dataset):
        raise NotImplementedError
    
    @abstractmethod
    def build_prompt(self, line, dataset):
        raise NotImplementedError
    