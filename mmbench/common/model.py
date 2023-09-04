import copy

from typing import Any
from sat.training.model_io import load_checkpoint


class ModelInterface(object):
    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 image_length=None,
                 text_processor=None,
                 text_processor_inference=None,
                 image_processor=None,
                 cross_image_processor=None,
                 **kwargs) -> None:
        self.args = copy.deepcopy(args)

        self.model = model
        self.tokenizer = tokenizer
        self.image_length = image_length
        self.text_processor = text_processor
        self.text_processor_inference = text_processor_inference
        self.image_processor = image_processor
        self.cross_image_processor = cross_image_processor

        # restore finetuned params
        self.finetuned_params = set()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.finetuned_params.add(n)
                
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)
    
    def reset_model(self):
        load_checkpoint(self.model, self.args)
        self.recover_model()
        

    def freezen_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad_(False)
    
    def recover_model(self):
        for n, p in self.model.named_parameters():
            if n in self.finetuned_params:
                p.requires_grad_(True)