'''
@File    :   base_task.py
@Time    :   2023/09
@Author  :   Wenmeng Yu
@Contact :   iyuge2@qq.com
'''
import os
import copy
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from functools import partial, wraps
from collections import defaultdict
from torch.nn import CrossEntropyLoss

from sat import mpu
from sat.helpers import print_rank0
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.data_utils.webds import MetaDistributedWebDataset

from mmbench.common.utils import find_all_files
from mmbench.common.inference import testing_main
from mmbench.common.training import training_main
from mmbench.dataset import ItemDataset, WdsDataset

class BaseTask(object):
    def __init__(self,
                 task_cfg,
                 custom_functions=dict(),
                 custom_dataset_functions=dict(),
                 image_length=None):
        self.task_cfg = task_cfg
        self.need_finetune = task_cfg.get('need_finetune', False)
        self.need_evaluate = task_cfg.get('need_evaluate', False)
        self.max_source_length = task_cfg["data_params"].get("max_source_length", 256) + image_length
        self.max_target_length = task_cfg["data_params"].get("max_target_length", 128) + image_length
        self.no_prompt = task_cfg["data_params"].get("no_prompt", False)

        self.custom_functions = custom_functions
        self.custom_dataset_functions = custom_dataset_functions

        self.dataloader_mirror = {}
        self.dataset_mirror = {}
    
    @NotImplementedError
    def calc_scores(self, args, results_total):
        pass

    def custom_func(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            if self.custom_functions.get(func.__name__, None):
                return self.custom_functions[func.__name__](*args, **kwargs)
            return func(self, *args, **kwargs)
        return new_func

    def partial_wo(self, func, *args, **kwargs):
        if self.custom_functions.get(func.__name__, None):
            return func
        return partial(func, *args, **kwargs)

    def fetch_dataset_mirror(self, args):
        """save all data to pd.DataFrame for computing metric scores
        """
        print_rank0(f'[{self.mode}]: fetch data mirror begin.')
        def get_data(dataloader):
            result = []
            top_keys, meta_keys = ["datatype", "question_id"], None
            for item in dataloader:
                qa = item["json"]
                if meta_keys is None:
                    meta_keys = list(qa["metadata"].keys())
                c_res = [qa[k] for k in top_keys] + [qa["metadata"][k] for k in meta_keys]
                result.append(c_res)
            return pd.DataFrame(result, columns=top_keys + meta_keys, dtype=str)

        if self.mode not in self.dataset_mirror:
            dataloader = self.dataloader_mirror["val"] if self.mode == "finetune" else self.dataloader_mirror["test"]
            self.dataset_mirror[self.mode] = get_data(dataloader)
        mirror_df = self.dataset_mirror[self.mode]
        print_rank0(f'fetch {self.mode} data mirror end.')
        return mirror_df

    def handle_upload_data(self, args, res_df):
        """convert test results to the format of eval website.

        Args:
            res_df (pd.DataFrame): two columns
                question_id (str): quesiton id
                preds (str): predict results
        """
        ret = [{"question_id": row["question_id"], "answer": row["preds"]} for idx, row in res_df.iterrows()]
        with open(args.save_details_result_path + ".json", "w") as fp:
            json.dump(ret, fp, indent=4)
        print_rank0(f'Save results: {args.save_details_result_path + ".json"}')

    def handle_metrics(self, args, results_total):
        question_ids, preds = results_total["question_ids"], results_total["preds"]
        res_df = pd.DataFrame({"question_id": question_ids, "preds": preds}, dtype=str)
        if self.mode == "upload":
            self.handle_upload_data(args, res_df)
            return {}
        before_res_len = len(res_df)
        # remove duplicates
        res_df = res_df.drop_duplicates(subset=["question_id"])
        if before_res_len != len(res_df):
            print_rank0(f"Sample nums change after removing duplicates: {before_res_len} -> {len(res_df)}", level=logging.WARNING)
        # get mirror data
        mirror_df = self.fetch_dataset_mirror(args)
        if len(res_df) != len(mirror_df):
            print_rank0(f"Sample nums not same: {len(res_df)} != {len(mirror_df)}", level=logging.WARNING)
        res_df = res_df.merge(mirror_df, on="question_id", how="inner")
        if self.mode == "test":
            res_df.to_csv(args.save_details_result_path, index=None)
        return self.calc_scores(args, res_df)

    def create_dataset_function(self, mt, path, args):
        path, data_mode = path.split("###")
        other_attr = self.other_attr if hasattr(self, "other_attr") else []
        item_dataset = ItemDataset(mt, args, path, data_mode, other_attr, custom_functions=self.custom_dataset_functions)
        if args.iterable_dataset:
            urls = find_all_files(path)
            web_dataset = WdsDataset(mt, args, data_mode, other_attr)
            dataset = MetaDistributedWebDataset(urls, web_dataset.process_fn_dataset, args.seed, meta_names=['json'])
            # create mirror dataset
            self.dataloader_mirror[data_mode] = item_dataset.data
        else:
            dataset = item_dataset
            self.dataloader_mirror[data_mode] = item_dataset.data
        return dataset
    
    def update_params(self, args, param_types: list):
        merge_args = copy.deepcopy(vars(args))
        for param_type in param_types:
            if hasattr(self.task_cfg, param_type):
                for key, value in self.task_cfg[param_type].items():
                    merge_args[key] = value
        if hasattr(self.task_cfg, "data"):
            for key, value in self.task_cfg.data.items():
                if isinstance(value, str):
                    value = [value]
                merge_args[key] = []
                for p in value:
                    tmp_path = os.path.join(args.data_home_dir, p)
                    merge_args[key].append(tmp_path)
        return argparse.Namespace(**merge_args)
    
    @custom_func
    def collate_fn(self, mt, examples):
        examples = [ex for ex in examples if len(ex) > 0] # drop {}
        if len(examples) == 0:
            return {}
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
            if len(data) == 0:
                return None
        else:
            return None
        timers('data loader').stop()
        data_b = self.broadcast_auto(data)
        for k in data_b:
            if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
                if args.fp16:
                    data_b[k] = data_b[k].half()
                elif args.bf16:
                    data_b[k] = data_b[k].bfloat16()
        return data_b

    @custom_func
    def forward_step(self, data_iterator, model, args, timers):
        # Get the batch.
        timers('batch generator').start()
        data_b = self.get_batch(
            data_iterator, args, timers)
        if data_b is None:
            return torch.tensor(0, device=args.device), {}
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

        return loss, {}

    @custom_func
    def preprocess_datab_eval(self, data_b):
        if isinstance(data_b["context_length"], int):
            context_len = data_b["context_length"]
        else:
            context_len = int(data_b['context_length'][0])
        tokens = data_b['input_ids'][:, :context_len]
        return data_b, tokens, context_len
    
    def chat(self, tokens, mt, args, **kwargs):
        if self.custom_functions.get("chat", None):
            return self.custom_functions["chat"](mt.model, mt.tokenizer, mt.text_processor_inference, tokens, args, **kwargs)
        inputs = tokens.to(mt.model.parameters().__next__().device)[0]
        seq = torch.cat(
            [inputs, torch.tensor([-1] * (self.max_source_length + self.max_target_length - len(inputs)), device=inputs.device)], dim=0
        )
        strategy = BaseStrategy(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, end_tokens=[mt.tokenizer.eos_token_id])
        # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
        #                               num_beams=num_beams, consider_end=True)
        get_func = mt.text_processor_inference.get_func(None, image_rope_mask=kwargs['image_rope_mask'])
        output = filling_sequence(
            mt.model, seq,
            batch_size=1,
            strategy=strategy,
            get_masks_and_position_ids=get_func,
            **kwargs
        )[0]  # drop memory

        return output

    @custom_func
    def forward_step_eval(self, mt, data_iterator, model, args, timers):
        # Get the batch.
        timers('batch generator').start()
        data_b = self.get_batch(
            data_iterator, args, timers)
        if data_b is None:
            return torch.tensor(0, device=args.device), {}
        timers('batch generator').stop()
        data_b, tokens, context_len = self.preprocess_datab_eval(data_b)
        question_id = data_b.pop('question_id')[0]
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        outputs = self.chat(tokens, mt, args, **data_b)[0][context_len:]
        model.del_mixin('auto-regressive')

        outputs = outputs.unsqueeze(0)
        pred = mt.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        metrics = {"question_ids": str(question_id), "preds": pred}
        return torch.tensor(0, device=outputs.device), metrics
    
    def do_finetune(self, args, mt):
        finetune_args = self.update_params(args, param_types=["finetune_params", "eval_params", "data_params"])
        if not ("train_data" in finetune_args and finetune_args.train_data):
            raise ValueError(f"[{self.task_name}]: train_data is required for finetuning.")
        # train
        self.mode = "finetune"
        finetune_args.mode = "finetune"
        training_main(finetune_args,
                      model_cls=mt.model,
                      forward_step_function=self.forward_step,
                      forward_step_eval=self.partial_wo(self.forward_step_eval, mt),
                      create_dataset_function=self.partial_wo(self.create_dataset_function, mt),
                      handle_metrics_function=self.partial_wo(self.handle_metrics, finetune_args),
                      collate_fn=self.partial_wo(self.collate_fn, mt))

    def do_evaluate(self, args, mt) -> dict:
        test_args = self.update_params(args, param_types=["eval_params", "data_params"])
        if not ("test_data" in test_args and test_args.test_data):
            raise ValueError(f"[{self.task_name}]: test_data is required for testing.")
        test_args.train_data = None
        test_args.valid_data = None
        # test
        self.mode = "test"
        test_args.mode = "inference"
        test_args.do_test = True

        # handle test data
        if hasattr(test_args, 'test_data') and test_args.test_data is not None:
            test_args.strict_eval = True
            if test_args.strict_eval and test_args.iterable_dataset:
                self.create_dataset_function(mt, test_args.test_data[0], test_args)
                test_args.eval_iters = len(self.dataloader_mirror["test"])
                test_args.strict_eval = False
                print_rank0(f'Due to strict_eval and iterable_dataset, resize eval_iters: \
                    {test_args.eval_iters}', level=logging.WARNING)
            # debug
            # test_args.strict_eval = False
            # test_args.eval_iters = 200
            # test_args.eval_interval = 1
            # debug
            test_args.load = test_args.save if self.need_finetune else None
            _, metrics = testing_main(test_args,
                                        model=mt.model,
                                        forward_step_eval=self.partial_wo(self.forward_step_eval, mt),
                                        create_dataset_function=self.partial_wo(self.create_dataset_function, mt),
                                        handle_metrics_function=self.partial_wo(self.handle_metrics, test_args),
                                        collate_fn=self.partial_wo(self.collate_fn, mt))

        # handle upload data
        if hasattr(test_args, 'upload_data') and test_args.upload_data:
            self.mode = "upload"
            print_rank0(f"Find {len(test_args.upload_data)} no-answer test datasets.")
            for upload_data in test_args.upload_data:
                print_rank0(f"Start {upload_data}...")
                test_args.test_data = [f'{upload_data}###upload']
                test_args.save_details_result_path = test_args.save_details_result_path.split('.')[0] + '-' +\
                                                        os.path.basename(upload_data)
                if test_args.iterable_dataset:
                    self.create_dataset_function(mt, test_args.test_data[0], test_args)
                    test_args.eval_iters = len(self.dataloader_mirror["upload"])
                    test_args.strict_eval = False
                    print_rank0(f'Due to strict_eval and iterable_dataset, resize eval_iters: \
                        {test_args.eval_iters}', level=logging.WARNING)
                # start
                testing_main(test_args,
                            model=mt.model,
                            forward_step_eval=self.partial_wo(self.forward_step_eval, mt),
                            create_dataset_function=self.partial_wo(self.create_dataset_function, mt),
                            handle_metrics_function=self.partial_wo(self.handle_metrics, test_args),
                            collate_fn=self.partial_wo(self.collate_fn, mt))
                print_rank0(f"End {upload_data}...")
        return metrics