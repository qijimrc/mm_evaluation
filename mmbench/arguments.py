import os
import torch
import argparse

from mmdoctor.common.logger import log

def add_logging_args(parser):
    group = parser.add_argument_group('logging')

    group.add_argument("--log_interval", type=int, default=10)
    group.add_argument("--save", type=str, default=None,
                       help="directory for saving all results")
    group.add_argument("--use_debug_mode", type=int, default=0)
    return parser

def add_inference_args(parser):
    group = parser.add_argument_group('inference')

    group.add_argument('--eval_tasks', type=str, nargs='+', default=None,
                       help='Specify the tasks for evaluation')
    group.add_argument('--eval_models', type=str, nargs='+', default=None,
                       help='Specify the models for evaluation')
    group.add_argument("--repetition_penalty", type=float, default=1.0,
                       help='repetition penalty, 1.0 means no penalty.')
    group.add_argument("--top_p", type=float, default=0.6, help='top p for nucleus sampling')
    group.add_argument("--top_k", type=int, default=2, help='top k for top k sampling')
    group.add_argument("--temperature", type=float, default=0.8, help='temperature for sampling')
    group.add_argument("--model_parallel_size", type=int, default=1, help='model parallel size used in inference')
    group.add_argument("--batch_size", type=int, default=1, help='batch size of loading data')
    group.add_argument("--num_workers", type=int, default=0, help='num of workers in dataloader')
    group.add_argument("--pad_noimg", action="store_true", help='add white image when processing pure text')
    return parser

def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='mmeval')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_inference_args(parser)
    parser = add_logging_args(parser)

    args = parser.parse_args(args_list)

    # initialize_backend
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.local_rank = int(os.getenv("LOCAL_RANK", '0')) # torchrun
   
    if torch.cuda.device_count() == 0:
        args.device = 'cpu'
    elif args.local_rank is not None:
        args.device = args.local_rank
    else:
        args.device = args.rank % torch.cuda.device_count()

    initialize_backend(args)

    return args

def get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
    return port

def initialize_backend(args):
    if torch.distributed.is_initialized():
        return True

    if args.device == 'cpu':
        pass
    else:
        torch.cuda.set_device(args.device)
    # Call the init process
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    if args.world_size == 1:
        from sat.helpers import get_free_port
        default_master_port = str(get_free_port())
    else:
        default_master_port = '6000'
    master_port = os.getenv('MASTER_PORT', default_master_port)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://{}:{}'.format(master_addr, master_port),
        world_size=args.world_size,
        rank=args.rank
    )
    return True