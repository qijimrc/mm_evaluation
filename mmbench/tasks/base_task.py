import os
import copy
import json
import torch
import random
import argparse
import numpy as np

from functools import partial, wraps
from collections import defaultdict
from torch.nn import CrossEntropyLoss

from sat import mpu, get_args
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.data_utils.webds import SimpleDistributedWebDataset

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.common.utils import get_tar_files
from mmbench.common.inference import testing_main
from mmbench.common.training import training_main
from typing import Any, Dict, List, Optional

class BaseTask(object):
    def __init__(self, task_cfg, custom_functions=dict(), **kwargs):
        self.task_cfg = task_cfg
        self.metrics = task_cfg.metrics
        self.need_finetune = task_cfg.get('need_finetune', False)
        self.custom_functions = custom_functions
    
    def custom_func(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            if self.custom_functions.get(func.__name__, None):
                return self.custom_functions[func.__name__](*args, **kwargs)
            return func(self, *args, **kwargs)
        return new_func

    @NotImplementedError
    def calc_scores(self, args, results_total):
        pass
    
    @NotImplementedError
    def create_dataset_function(self, mt, path, args):
        pass

    def update_params(self, args, param_type="eval_params"):
        merge_args = copy.deepcopy(vars(args))
        if hasattr(self.task_cfg, param_type):
            for key, value in self.task_cfg[param_type].items():
                merge_args[key] = value
        if hasattr(self.task_cfg, "data"):
            for key, value in self.task_cfg.data.items():
                tmp_path = os.path.join(args.data_home_dir, value)
                assert os.path.exists(tmp_path), f"cannot found {tmp_path}"
                merge_args[key] = [tmp_path]
        return argparse.Namespace(**merge_args)

    def data_processor(self, urls, args, mt, **kwargs):
        return SimpleDistributedWebDataset(urls, partial(self.process_fn_dataset, args, mt), args.seed)
      
    def data_collator(self, mt, examples):
        for example in examples:
            for k in example:
                if isinstance(example[k], list):
                    example[k] = torch.tensor(example[k])
                elif isinstance(example[k], np.ndarray):
                    example[k] = torch.from_numpy(example[k])
        img_args = {}
        tmp_example = examples[0]
        for k in tmp_example['vision']:
            if type(tmp_example['vision'][k]) is torch.Tensor:
                img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
            else:
                img_args['vision_'+k] = example['vision'][k]
        if mt.cross_image_processor is not None:
            img_args.update({'cross_'+k: torch.cat([example['cross'][k] for example in examples]) if type(example['cross'][k]) is torch.Tensor else example['cross'][k] for k in example['cross']})
        for example in examples:
            example.pop('vision')
            if 'cross' in example:
                example.pop('cross')

        model_args = {}
        tmp_example = examples[0]
        for k in tmp_example:
            if type(tmp_example[k]) is torch.Tensor:
                model_args[k] = torch.cat([example[k] for example in examples])
            elif isinstance(tmp_example[k], str):
                model_args[k] = [example[k] for example in examples]
            else:
                model_args[k] = tmp_example[k]
        model_args.update(img_args)
        return model_args

    def broadcast_auto(self, data_dict):
        type2list = defaultdict(list)
        other = []
        for k in data_dict:
            if type(data_dict[k]) is torch.Tensor:
                type2list[data_dict[k].dtype].append(k)
            else:
                other.append(k)
        new_data = {}
        for k in type2list:
            new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
        for k in other:
            new_data[k] = data_dict[k]
        return new_data
    
    def get_batch(self, data_iterator, args, timers):
        # Broadcast data.
        timers('data loader').start()
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        timers('data loader').stop()
        data_b = self.broadcast_auto(data)
        for k in data_b:
            if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
                if args.fp16:
                    data_b[k] = data_b[k].half()
                elif args.bf16:
                    data_b[k] = data_b[k].bfloat16()
        return data_b

    def forward_step(self, data_iterator, model, args, timers):
        if self.custom_functions.get("forward_step", None):
            return self.custom_functions["forward_step"](data_iterator, model, args, timers)
        # Get the batch.
        timers('batch generator').start()
        data_b = self.get_batch(
            data_iterator, args, timers)
        labels = data_b.pop('labels')
        timers('batch generator').stop()
        logits = model(**data_b)[0]
        lm_logits = logits.to(torch.float32)
        # Shift so that tokens < n predict n
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.to(torch.float32)

        return loss, {'loss': loss}
    
    def chat(self, model, tokenizer, text_processor_inference, tokens,
         max_length: int = 1300, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
        inputs = tokens.to(model.parameters().__next__().device)[0]
        seq = torch.cat(
            [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
        )
        strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
        # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
        #                               num_beams=num_beams, consider_end=True)
        get_func = text_processor_inference.get_func(None, image_rope_mask=kwargs['image_rope_mask'])
        output = filling_sequence(
            model, seq,
            batch_size=1,
            strategy=strategy,
            get_masks_and_position_ids=get_func,
            **kwargs
        )[0]  # drop memory

        return output

    def precess_datab_in_eval(self, context_len, data_b):
        if self.custom_functions.get("precess_datab_in_eval", None):
            return self.custom_functions["precess_datab_in_eval"](context_len, data_b)
        data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
        data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
        data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

        data_b.pop('input_ids')
        data_b.pop('attention_mask')
        data_b.pop('position_ids')
        return data_b
    
    def forward_step_eval(self, mt, data_iterator, model, args, timers):
        if self.custom_functions.get("forward_step_eval", None):
            return self.custom_functions["forward_step_eval"](data_iterator, model, args, timers)
        # Get the batch.
        timers('batch generator').start()
        data_b = self.get_batch(
            data_iterator, args, timers)
        timers('batch generator').stop()

        context_len = int(data_b['context_length'][0])
        tokens = data_b['input_ids'][:, :context_len]
        data_b = self.precess_datab_in_eval(context_len, data_b)
        label_text = data_b.pop('label_text')
        question_id = data_b.pop('question_id')

        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        outputs = self.chat(model, mt.tokenizer, mt.text_processor_inference, tokens, **data_b)[0][context_len:]
        model.del_mixin('auto-regressive')

        outputs = outputs.unsqueeze(0)
        pred = mt.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        metrics = {"question_ids": str(question_id), "labels": label_text[0], "preds": pred}
        return torch.tensor(0, device=outputs.device), metrics
    
    def do_finetune(self, args, mt):
        self.mode = "finetune"
        finetune_args = self.update_params(args, param_type="finetune_params")
        if 'experiment_name' not in finetune_args:
            finetune_args.experiment_name = f'{self.task_name}'
        else:
            finetune_args.experiment_name = f'{args.experiment_name}_{self.task_name}'
        if not ("train_data" in finetune_args and finetune_args.train_data):
            raise ValueError(f"[{self.task_name}]: train_data is required for finetuning.")
        # train
        training_main(finetune_args,
                      model_cls=mt.model,
                      forward_step_function=self.forward_step,
                      forward_step_eval=partial(self.forward_step_eval, mt),
                      create_dataset_function=partial(self.create_dataset_function, mt),
                      handle_metrics_function=partial(self.calc_scores, finetune_args),
                      collate_fn=partial(self.data_collator, mt))
    
    def do_evaluate(self, args, mt) -> dict:
        self.mode = "test"
        test_args = self.update_params(args, param_type="eval_params")
        # test
        loss, metrics = testing_main(test_args,
                                     model=mt.model,
                                     forward_step_eval=partial(self.forward_step_eval, mt),
                                     create_dataset_function=partial(self.create_dataset_function, mt),
                                     handle_metrics_function=partial(self.calc_scores, test_args),
                                     collate_fn=partial(self.data_collator, mt))
        metrics["total_loss"] = loss
        return metrics