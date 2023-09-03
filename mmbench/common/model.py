
from typing import Any


class ModelInterface(object):
    def __init__(self,
                 model,
                 tokenizer,
                 image_length=None,
                 text_processor=None,
                 text_processor_inference=None,
                 image_processor=None,
                 cross_image_processor=None,
                 **kwargs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_length = image_length
        self.text_processor = text_processor
        self.text_processor_inference = text_processor_inference
        self.image_processor = image_processor
        self.cross_image_processor = cross_image_processor
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)