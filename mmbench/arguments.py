import argparse

def add_inference_args(parser):
    group = parser.add_argument_group('inference')

    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--repetition_penalty", type=float, default=1.0,
                       help='repetition penalty, 1.0 means no penalty.')
    group.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    group.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    group.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    group.add_argument("--model_parallel_size", type=int, default=1, help='model parallel size used in inference')

    group.add_argument("--log_interval", type=int, default=10)
    group.add_argument("--save", type=str, default=None)
    group.add_argument("--model_cache_dir", type=str, default=None, 
                       help="dir caching model ckpts, like huggingface, modelscope, ...")
    group.add_argument("--use_debug_mode", type=bool, default=False)
    return parser

def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='sat')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_inference_args(parser)
    args = parser.parse_args(args_list)
    return args