import torch
from functools import partial
from sat.training.deepspeed_training import training_main

from mmbench.common.registry import Registry
from mmbench.common.example import Example
from mmbench.utils.utils import get_tar_files
from typing import Any, Dict, List, Optional

class BaseTask:
    def __init__(self):
        pass

    def __call__(self, index=None) -> Example:
        """ Default iterator to return current example of given index.
        """
        if index:
            if index >= len(self.examples): raise StopIteration
            ex = self.examples[index]
        else:
            if self.cur >= len(self.examples): raise StopIteration
            ex = self.examples[self.cur]
            self.cur += 1
        return ex

    def __len__(self) -> int:
        return len(self.examples)
      
    def _data_collator(self, examples):
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
            else:
                model_args[k] = tmp_example[k]
        model_args.update(img_args)
        return model_args

    def _forward_step(self):
        # TODO
        pass
    
    def _forward_step_eval(self):
        # TODO
        pass
    
    def _calc_scores(self):
        # TODO
        pass
      
    def _data_processor(self, path, args):
        """ Default data processor
        """
        # TODO
        pass
    
    def _create_dataset_function(self, model, data_processor, tokenizer, path, args):
        txt_processor = mt.text_processor(tokenizer, args.max_source_length+args.max_target_length, image_length=mt.image_length, model=model)
        urls = get_tar_files(path)
        if args.random_urls:
            random.shuffle(urls)
            print(urls[:10])
        dataset = data_processor(urls, args, mt.image_processor, txt_processor)
        return dataset

    def do_finetune(self,
                    args,
                    model,
                    tokenizer,
                    text_processor,
                    img_processor):
        # TODO: 构建好 finetune 需要的东西
        training_main(args,
                      model_cls=model,
                      forward_step_function=self._forward_step,
                      create_dataset_function=partial(self._create_dataset_function, model, self._data_processor, tokenizer),
                      collate_fn=self._data_collator)
    
    def do_evaluate(self) -> dict:
        # 参考 training main，完成 evaluate 的基本逻辑
        scores = evaluating_main()