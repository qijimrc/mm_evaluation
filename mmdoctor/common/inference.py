import torch
import logging
import numpy as np

from mmdoctor.common.logger import log

from .global_vars import *

def inference_main(args, dataloader, model_cls):
    rank = args.rank
    eval_iters = len(dataloader)
    ret_total = {}
    with torch.no_grad():
        iteration = 0
        for data in dataloader:
            iteration += 1
            sync_ret = {"question_ids": data[0]["question_id"]}
            try:
                sync_ret["preds"] = model_cls.generate(prompt=data[0]["question"],
                                                    history=data[0]["history"],
                                                    image_path=data[0]["image_path"])
            except:
                sync_ret["preds"] = PAD_STR
            for name, value in sync_ret.items():
                # print_all(f"{name}: {value}")
                if name not in ret_total:
                    ret_total[name] = []
                if len(value) == 0:
                    value = PAD_STR
                byte_value = value.encode('utf-8')
                byte_tensor = torch.tensor(bytearray(byte_value),
                                           dtype=torch.uint8,
                                           device="cuda")
                # Gathers tensor arrays of different lengths across multiple gpus
                byte_list = all_gather(byte_tensor, args.world_size)
                
                if rank == 0:
                    gathered_len = len(byte_list)
                    for i in range(gathered_len):
                        decode_bytes = np.array(byte_list[i].cpu()).tobytes()
                        try:
                            decode_value = decode_bytes.decode('utf-8')
                        except Exception as e1:
                            try:
                                decode_value = decode_bytes.decode('ISO-8859-1')
                            except Exception as e2:
                                decode_value = DECODE_ERROR_STR
                                log(f'Decode failed, the output is replaced by {decode_value}.', level=logging.ERROR)
                        ret_total[name].append(decode_value)
            if iteration % args.log_interval == 0:
                log('Evaluating iter {}/{}'.format(iteration, eval_iters))
    return ret_total
    
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
