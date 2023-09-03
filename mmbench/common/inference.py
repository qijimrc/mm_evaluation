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

import os
import random
import math
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from contextlib import ExitStack

import deepspeed

from sat.training.learning_rates import AnnealingLR
from sat.training.model_io import load_checkpoint, save_checkpoint
from sat.training.utils import Timers
from sat.training.utils import report_memory
from sat.training.utils import print_args
from sat.training.utils import get_sample_writer

from sat import mpu
from sat.data_utils import make_loaders
from sat.ops.layernorm import LayerNorm
from sat.helpers import print_rank0, print_all
from sat.model.base_model import get_model

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

    args.experiment_name = args.experiment_name + '-' +datetime.now().strftime("%m-%d-%H-%M")

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'], collate_fn=collate_fn)

    # Config model IO
    if args.load is not None:
        args.iteration = load_checkpoint(model, args)
        # if we don't load optim_states, filelock is no more needed.
        # with FileLock("/root/checkpoint_lock", timeout=-1):
        #     args.iteration = load_checkpoint(model, optimizer, args)
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)

    torch.distributed.barrier()

    # final testing
    if args.do_test and test_data is not None:
        prefix = 'test data'
        test_loss, metrics = evaluate_and_print_results(prefix, iter(test_data),
            model, len(test_data) if args.strict_eval else args.eval_iters, args, timers, True, split='test', hooks=hooks)
        return test_loss, metrics
    return None, None

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
    is_scalar = {}
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
            for name in metrics:
                if name not in metrics_total:
                    metrics_total[name] = []
                is_scalar[name] = True if len(metrics[name].shape)==0 else False
                shape = list(metrics[name].shape)
                if not is_scalar[name] and is_last and metrics[name].shape[0] != last_shape[0]:
                    # pad tensor's first dim to args.batch_size
                    metrics[name] = torch.concat([metrics[name], torch.zeros([last_shape[0]-metrics[name].shape[0]] + shape[1:], dtype=metrics[name].dtype, device=metrics[name].device)])
                if rank==0:
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                else:
                    # metrics_gathered = None
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                # torch.distributed.gather(metrics[name], metrics_gathered, 0)
                torch.distributed.all_gather(metrics_gathered, metrics[name])

                if rank==0:
                    gathered_len = len(metrics_gathered) if not is_last else len(metrics_gathered) - drop_number * args.model_parallel_size
                    for i in range(gathered_len):
                        if is_scalar[name] or not is_last:
                            metrics_total[name].append(metrics_gathered[i].data.cpu())
                        else:
                            metrics_total[name].append(metrics_gathered[i][:last_shape[i]].data.cpu())
    total_lm_loss /= eval_iters
    # metrics_avg = {key: value / eval_iters for key, value in metrics_total.items()}
    if rank==0:
        for name in metrics_total:
            if is_scalar[name]:
                metrics_total[name] = torch.stack(metrics_total[name], dim=0)
            else:
                metrics_total[name] = torch.concat(metrics_total[name], dim=0)
        if hooks['handle_metrics'] is not None:
            metrics = hooks['handle_metrics'](metrics_total)
        else:
            for name in metrics_total:
                assert is_scalar[name], 'you must return scalar metrics or implement handle_metrics hooks'
            metrics = {key: sum(value.split(1,0))/len(value) for key, value in metrics_total.items()}
    else:
        metrics = None
    return total_lm_loss, metrics

def evaluate_and_print_results(prefix, data_iterator, model, eval_iters,
                            args, timers, has_last, split, verbose=False, step=None, summary_writer=None, hooks={}):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, metrics = evaluate(data_iterator, model, eval_iters, args, timers, split, verbose, has_last, hooks=hooks)
    lm_ppl = math.exp(min(20, lm_loss))
    if torch.distributed.get_rank(group=mpu.get_data_parallel_group())==0:
        report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step, metrics)
    return lm_loss, metrics


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args, avg_metrics):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' total loss {:.6E} |'.format(loss)
    for key in avg_metrics:
        log_string += ' {} {:.6E} |'.format(key, avg_metrics[key])
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    log_string += 'speed {:.2f} samples/(min*GPU)'.format(
        (args.gradient_accumulation_steps * args.batch_size / args.model_parallel_size / (elapsed_time / 60000.0)))
    print_rank0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/'+key, avg_metrics[key], step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step, avg_metrics):
    string = ' validation loss at {} | '.format(prefix)
    string += 'loss: {:.6E} | '.format(loss)
    string += 'PPL: {:.6E}'.format(ppl)
    for key in avg_metrics:
        string += ' {} {:.6E} |'.format(key, avg_metrics[key].item())
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
        