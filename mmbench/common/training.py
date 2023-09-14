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
from sat.training.deepspeed_training import setup_model_untrainable_params_and_optimizer, get_learning_rate_scheduler, train


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
