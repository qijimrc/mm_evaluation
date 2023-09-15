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
from sat.training.deepspeed_training import setup_model_untrainable_params_and_optimizer, get_learning_rate_scheduler

from .inference import evaluate_and_print_results

def training_main(args, model_cls, forward_step_function, create_dataset_function, handle_metrics_function=None, init_function=None, collate_fn=None, forward_step_eval=None):
    """Main training program."""
    hooks = {
        'forward_step': forward_step_function,
        'init_function': init_function,
        'create_dataset_function': create_dataset_function,
        'handle_metrics': handle_metrics_function,
        'forward_step_eval': forward_step_eval or forward_step_function
    }

    timers = Timers()  # Timer.

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'], collate_fn=collate_fn)
    if args.epochs:
        args.train_iters = len(train_data)
        if args.eval_interval is None:
            args.eval_interval = len(train_data)//args.epochs
        if args.save_interval is None:
            args.save_interval = len(train_data)//args.epochs

    # Build model
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    
    # Config model IO
    if args.load is not None:
        args.iteration = load_checkpoint(model, args)
    else:
        args.iteration = 0

    torch.distributed.barrier()

    # init hook before building deepspeed model and optimizer
    if hooks['init_function'] is not None:
        hooks['init_function'](args, model)

    # Optimization related things
    model, optimizer = setup_model_untrainable_params_and_optimizer(args, model)

    # initialize lr scheduler
    lr_scheduler = get_learning_rate_scheduler(optimizer, args.iteration, args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        if args.mode == 'pretrain':
            print_rank0('Pretraining or Continuing training the Model...')
        elif args.mode == 'finetune':
            print_rank0('Finetuning Model...')
        print_args(args)
        summary_writer = get_sample_writer(base=args.summary_dir, name=args.experiment_name, iteration=args.iteration)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if not args.iterable_dataset:
            if train_data is not None:
                train_data.batch_sampler.start_iter = args.iteration % len(train_data)
            if val_data is not None:
                start_iter_val = (args.train_iters // args.save_interval) * args.eval_interval
                val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        else:
            print_rank0('Warning: we cannot resume iterable dataloader. skipping...')

    # training 
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)
                iteration, skipped = train(model, optimizer,
                    lr_scheduler,
                    train_data,
                    val_data,
                    timers, args, summary_writer=summary_writer,
                    hooks=hooks
                    )

    # final save
    if args.save and iteration != 0:  # TODO save
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

def train(model, optimizer, lr_scheduler,
        train_data, val_data, timers, args, 
        summary_writer=None, hooks={}):
    """Train the model."""
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
        
    # Turn on training mode which enables dropout.
    model.train()
    
    # Tracking loss.
    total_lm_loss = 0.0
    total_metrics = defaultdict(float)

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, metrics = train_step(train_data_iterator,
                                                    model,
                                                    optimizer,
                                                    lr_scheduler,
                                                    args, timers, hooks=hooks)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        for name in metrics:
            if not 'eval' in name:
                assert len(metrics[name].shape)==0, 'metrics without eval must be scalar'
                total_metrics[name] += metrics[name].data.detach().float().item()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            # average img & txt loss
            avg_metrics = {}
            for key in total_metrics:
                avg_metrics[key] = total_metrics[key] / args.log_interval

            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args,
                                     avg_metrics)
            total_lm_loss = 0.0
            total_metrics = defaultdict(float)
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False

            timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                        'batch generator', 'data loader'],
                       normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            if args.strict_eval:
                val_data_iterator = iter(val_data)
                eval_iters = len(val_data)
            else:
                eval_iters = args.eval_iters
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, eval_iters, args, timers, False, step=args.iteration, split='val', summary_writer=summary_writer, hooks=hooks)

        if args.exit_interval and args.iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_all('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, args.iteration), flush=True)
            exit()

    return args.iteration, skipped_iters


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, hooks=None, single_step=False, **kwargs):
    """Single training step."""
    if hooks is None:
        hooks = {}
    lm_loss_total, metrics_total, count = 0.0, {}, 0
    forward_step = hooks['forward_step']

    while True:
        # Forward model for one step.
        timers('forward').start()
        forward_ret = forward_step(data_iterator, model, args, timers, **kwargs)
        if isinstance(forward_ret, tuple):
            lm_loss, metrics = forward_ret
        else:
            lm_loss, metrics = forward_ret, {}
        timers('forward').stop()

        # Check nan or inf in forward, preventing it from interfering loss scaler,
        # and all reduce metrics by the way
        lm_loss_reduced = lm_loss.detach().clone()
        torch.distributed.all_reduce(lm_loss_reduced.data)
        lm_loss_reduced.data = lm_loss_reduced.data / args.world_size

        loss_checker = lm_loss_reduced
        for name in metrics:
            if not 'eval' in name:
                metrics[name] = metrics[name].detach().clone()
                torch.distributed.all_reduce(metrics[name].data)
                metrics[name].data /= args.world_size
                loss_checker = loss_checker + metrics[name]
        if loss_checker.isnan().any() or loss_checker.isinf().any():
            print_all('Skipping backward and optimizer step for nan or inf in forwarding metrics/loss!')
            return lm_loss.detach(), 1, metrics

        # Accumulate the statistics
        lm_loss_total += lm_loss_reduced
        for name in metrics:
            if name not in metrics_total:
                metrics_total[name] = 0.0
            metrics_total[name] += metrics[name]
        count += 1
        # Calculate gradients, reduce across processes, and clip.
        timers('backward').start()
        backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()
        # Update parameters.
        skipped_iter, complete = 0, False
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
        timers('optimizer').stop()
        if complete or single_step:
            break
    lm_loss_total /= count
    metrics_total = {key: value / count for key, value in metrics_total.items()}
    return lm_loss_total, skipped_iter, metrics_total


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError('Currently, we only support training with deepspeed.')

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()

    return

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