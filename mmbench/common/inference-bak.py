# coding=utf-8
# Rewrite by Ming Ding, Tsinghua University
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import logging
import deepspeed
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader

from sat import mpu
from sat.training.model_io import load_checkpoint
from sat.training.utils import Timers
from sat.data_utils import make_loaders
from sat.helpers import print_rank0, print_all

from .global_vars import *

def testing_main(args,
                 model,
                 create_dataset_function,
                 handle_metrics_function=None,
                 collate_fn=None,
                 forward_step_eval=None):
    """Main testing program."""
    hooks = {
        'create_dataset_function': create_dataset_function,
        'handle_metrics': handle_metrics_function,
        'forward_step_eval': forward_step_eval
    }

    timers = Timers()  # Timer.

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'], collate_fn=collate_fn)

    # Config model IO
    if args.load is not None:
        args.iteration = load_checkpoint(model, args)
    
    torch.distributed.barrier()

    # final testing
    if args.do_test and test_data is not None:
        prefix = 'test data'
        test_loss, metrics = evaluate_and_print_results(prefix, iter(test_data),
            model, len(test_data) if args.strict_eval else args.eval_iters, args, timers, True, split='test', verbose=True, hooks=hooks)
        return test_loss, metrics
    return 0.0, {} 

def evaluate_and_print_results(prefix, data_iterator, model, eval_iters,
                            args, timers, has_last, split, verbose=False, step=None, summary_writer=None, hooks={}):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, metrics = evaluate(data_iterator, model, eval_iters, args, timers, split, verbose, has_last, hooks=hooks)
    lm_ppl = math.exp(min(20, lm_loss))
    if torch.distributed.get_rank(group=mpu.get_data_parallel_group())==0:
        report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step, metrics)
    return lm_loss, metrics

def evaluate(data_iterator, model, eval_iters, args, timers, split, verbose=False, has_last=True, hooks={}):
    """Evaluation."""
    forward_step = hooks['forward_step_eval']
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    total_lm_loss, metrics_total = 0, {}
    if split=='val':
        last_shape = args.val_last_shape
        drop_number = args.val_drop_number
    else:
        assert split=='test'
        last_shape = args.test_last_shape
        drop_number = args.test_drop_number
    with torch.no_grad():
        iteration = 0
        while iteration < eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank0('Evaluating iter {}/{}'.format(iteration, eval_iters))
            # Forward evaluation.
            # try:
            lm_loss, metrics = forward_step(data_iterator, model, args, timers)
            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            total_lm_loss += lm_loss.data.detach().float().item()
            is_last = True if iteration == eval_iters and args.strict_eval and len(last_shape)>0 else False
            for name, value in metrics.items():
                # print_all(f"{name}: {value}")
                if name not in metrics_total:
                    metrics_total[name] = []
                if len(value) == 0:
                    value = PAD_STR
                byte_value = value.encode('utf-8')
                byte_tensor = torch.tensor(bytearray(byte_value), dtype=torch.uint8, device=lm_loss.device)
                # Gathers tensor arrays of different lengths across multiple gpus
                byte_list = all_gather(byte_tensor, args.world_size)
                
                if rank == 0:
                    gathered_len = len(byte_list) if not is_last else len(byte_list) - drop_number * args.model_parallel_size
                    for i in range(gathered_len):
                        decode_bytes = np.array(byte_list[i].cpu()).tobytes()
                        try:
                            decode_value = decode_bytes.decode('utf-8')
                        except Exception as e1:
                            try:
                                decode_value = decode_bytes.decode('ISO-8859-1')
                            except Exception as e2:
                                decode_value = DECODE_ERROR_STR
                                print_rank0(f'decode failed, the output is replaced by {decode_value}.', level=logging.ERROR)
                        metrics_total[name].append(decode_value)

    if split == "val":
        model.train()
    # Move model back to the train mode.
    total_lm_loss /= eval_iters
    # metrics_avg = {key: value / eval_iters for key, value in metrics_total.items()}
    if rank==0:
        if hooks['handle_metrics'] is not None:
            metrics = hooks['handle_metrics'](metrics_total)
        else:
            raise NotImplemented('custom handle_metrics is necessary in current.')
    else:
        metrics = None
    return total_lm_loss, metrics

def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step, avg_metrics):
    string = ' validation loss at {} | '.format(prefix)
    string += 'loss: {:.6E} | '.format(loss)
    string += 'PPL: {:.6E}'.format(ppl)
    for key in avg_metrics:
        string += ' {} {:.6E} |'.format(key, avg_metrics[key]) if type(avg_metrics[key]) is not dict else " {} {} |".format(key, str({k:round(v, 6) for k,v in avg_metrics[key].items()}))
    length = len(string) + 1
    print_rank0('-' * 100)
    print_rank0('-' * length)
    print_rank0(string)
    print_rank0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/valid_'+key, avg_metrics[key], step)
        
def all_gather(tensor, world_size):
    """Gathers tensor arrays of different lengths across multiple gpus
    """
    # gather all tensor size
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)
    # padding
    size_diff = max_size.item() - local_size.item()
    if size_diff:
        tensor = torch.cat([tensor, torch.zeros(size_diff, device=tensor.device, dtype=tensor.dtype)])
    all_tensor_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensor_padded, tensor)
    # un-padding
    ret = []
    for q, size in zip(all_tensor_padded, all_sizes):
        ret.append(q[:size])
    return ret