
import os
import re
import sys
import torch
import argparse

if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from functools import partial
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sat.model import AutoModel
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

from mmbench.evaluator import Evaluator
from mmbench.common.utils import is_chinese
from mmbench.common.model import ModelInterface
    
def _history_to_prompt(self, history, query, add_eoi_first=False):
    prompt = self.tokenizer.eoi if add_eoi_first else ""
    if not is_chinese(query):
        for i, (old_query, response) in enumerate(history):
            prompt += "Q:{}\nA:{}\n".format(old_query, response)
        prompt += "Q:{}\nA:".format(query)
    else:
        for i, (old_query, response) in enumerate(history):
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)
    return prompt
    
class chatglm_text_processor:
    def __init__(self, tokenizer, max_target_length=1024, image_length=257, model=None):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        self.model = model

    def __call__(self, caption, prompt=""):
        caption = self.pre_caption(caption)
        # print(prompt,caption)
        input0 = self.tokenizer.encode(self.tokenizer.boi, add_special_tokens=False)
        input1 = [self.tokenizer.pad_token_id] * self.image_length
        input2 = self.tokenizer.encode(prompt, add_special_tokens=False)

        a_ids = sum([input0, input1, input2], [])
        if len(a_ids) > self.max_target_length-3:
            return None

        b_ids = self.tokenizer.encode(text=caption, add_special_tokens=False)
        if len(a_ids) + len(b_ids) > self.max_target_length-3:
            b_ids = b_ids[: self.max_target_length-len(a_ids)-3]
        pre_image = len(input0)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position + 1:]

        pad_len = self.max_target_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0)
        attention_mask, position_ids = self.model.get_inputs(input_ids)

        return {'input_ids': input_ids, 'labels': labels, 'position_ids': position_ids, 'attention_mask': attention_mask, 'pre_image': pre_image}
    
    def history_to_prompt(self, history, query, add_eoi_first=False):
        return _history_to_prompt(self, history, query, add_eoi_first)

    def pre_caption(self, caption):
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption
    
class chatglm_text_processor_inference:
    def __init__(self, tokenizer, max_target_length=1024, image_length=257, model=None, no_prompt=False, english=False):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        sep = "A:" if english else "答："
        self.sep = sep if not no_prompt else self.tokenizer.eoi
        self.english = english
        self.invalid_slices = [slice(63823, 130000)] if english else []

    def __call__(self, prompt):
        input0 = self.tokenizer.encode(self.tokenizer.boi, add_special_tokens=False)
        input1 = [self.tokenizer.pad_token_id] * self.image_length
        input2 = self.tokenizer.encode(prompt, add_special_tokens=False)

        a_ids = sum([input0, input1, input2], [])
        pre_image = len(input0)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids)

        return {'input_ids': torch.tensor(input_ids).unsqueeze(0), 'pre_image': pre_image}

    def history_to_prompt(self, history, query, add_eoi_first=False):
        return _history_to_prompt(self, history, query, add_eoi_first)

class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

def blip2_image_processor_func(image_processor, image):
    return {'image': image_processor(image).unsqueeze(0)}

def load_model(args):
    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=args.rank,
        rank=args.rank,
        world_size=args.world_size,
        mode='inference',
        checkpoint_activations=None,
        checkpoint_num_layers=None,
        fp16=args.fp16,
        bf16=args.bf16,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
    ))
    if args.transfer:
        model_args.force_inference = True
        from sat.training.model_io import load_checkpoint
        load_checkpoint(model, model_args, args.transfer)
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert args.world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    if args.quant:
        quantize(model.transformer, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    return model

def chatglm_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.boi = "<img>"
    tokenizer.eoi = "</img>"
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, help='pretrained ckpt')
    parser.add_argument('--eval_tasks', type=str, nargs='+', help='Specify the tasks for evaluation')
    parser.add_argument("--output_filemark", type=str, default=None, help="name marks for output file")
    parser.add_argument("--max_length", type=int, default=1024, help='max length')
    parser.add_argument("--no_prompt", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=float, default=100, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--transfer", type=str, default="", help="train with a language model and inference with another language model")
    args = parser.parse_args()
    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    model = load_model(args)
    tokenizer = chatglm_tokenizer("/nxchinamobile2/shared/official_pretrains/sat_home/chatglm-6b")
    blip2_image_processor = partial(blip2_image_processor_func, BlipImageEvalProcessor(224))
    mt = ModelInterface(model, tokenizer, 32, chatglm_text_processor, chatglm_text_processor_inference, blip2_image_processor)
    
    evaluator = Evaluator()
    scores = evaluator.evaluate(mt, eval_tasks=["ScienceVQA"])