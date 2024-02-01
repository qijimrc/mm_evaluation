import torch
from PIL import Image

import sys
from mmbench.models.utils import osp, load, dump, timer
from mmbench.models.base_model import BaseModel
from mmbench.common.registry import Registry

from PIL import Image

from .yi_llava.mm_utils import get_model_name_from_path, load_pretrained_model
from .yi_llava.model.constants import key_info
from .yi_llava.conversation import conv_templates
from .yi_llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .yi_llava.mm_utils import KeywordsStoppingCriteria, expand2square, tokenizer_image_token

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def edit_config(repo_id):
    # manually removed added_tokens.json from 01ai/Yi-VL-6B 
    root = repo_id
    assert root is not None and osp.exists(root)
    cfg = osp.join(root, 'config.json')
    data = load(cfg)
    mm_vision_tower = data['mm_vision_tower']
    if mm_vision_tower.startswith('./vit/'):
        data['mm_vision_tower'] = osp.join(root, mm_vision_tower)
        assert osp.exists(data['mm_vision_tower'])
        dump(data, cfg)



@Registry.register_model('YiVL_6B')
@Registry.register_model('YiVL_34B')
class YiVLWrapper(BaseModel):
    
    @timer('init')
    def __init__(self, cfg, args,
                 **kwargs) -> None:
        model_name = cfg.model_name
        model_path = osp.join(args.model_cache_dir, model_name)
        
        if not osp.exists(model_path):
            raise ValueError(f"Please download Yi-VL model.")
        elif osp.exists(model_path):
            edit_config(model_path)

        disable_torch_init()
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            device_map='cpu')
        self.model = self.model.cuda()
        
        kwargs_default = dict(temperature=0.2,
                              num_beams= 1,
                              conv_mode= "mm_default",
                              top_p= None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
    
    @timer('generate')
    def generate(self, image_path, prompt, history=[]):
        
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.kwargs['conv_mode']].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
            )
        
        image = Image.open(image_path)
        if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
            if image.mode == 'L':
                background_color = int(sum([int(x * 255) for x in self.image_processor.image_mean]) / 3)
            else:
                background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, background_color)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
            ][0]
        
        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        self.model = self.model.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=self.kwargs['temperature'],
                top_p=self.kwargs['top_p'],
                num_beams=self.kwargs['num_beams'],
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
                )
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

if __name__ == '__main__':
    from pdb import set_trace as st
    model = YiVL_34B()
    resp = model.generate('/share/home/chengyean/evaluation/data/dummy_example/image.png', 
                          prompt='What does this image imply?',)
    print("*******")
    print(resp)

    
        
    