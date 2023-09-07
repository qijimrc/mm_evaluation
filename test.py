import io
import os
import json
import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt
import webdataset as wds

from PIL import Image
from glob import glob
from braceexpand import braceexpand
from torch.utils.data import DataLoader
from sat.data_utils.datasets import MetaDistributedWebDataset
from sat.helpers import print_rank0

def find_all_files(path):
    pathes = path.split(',')
    tar_files = []
    for p in pathes:
        if '*' in p:
            include_dirs, n = p.split('*')
            repeat_nums = int(n)
        else:
            include_dirs = p
            repeat_nums = 1
        if include_dirs.endswith('.tar'): # path/to/name-{000000..000024}.tar
            tar_files.extend(list(braceexpand(include_dirs)) * repeat_nums)
        else: # path/to/dataset_name
            for cur_dir, _, files in os.walk(include_dirs, followlinks=True):
                for f in files:
                    if f.endswith('.tar'):
                        tar_files.extend([os.path.join(cur_dir,f)]*repeat_nums)
    print(f'find {len(tar_files)} tars in all...')
    return tar_files

def process_fn(src):
    for data in src:
        ret = {
            "__key__": data["__key__"],
            "image": data["jpg"],
            "json": data["json"]
        }
        yield ret

def collate_fn(examples):
    ret = {
        k: [ex[k] for ex in examples] for k,v in examples[0].items()
    }
    return ret
        
import logging
print_rank0("test info", level=logging.INFO)
print_rank0("test warning", level=logging.WARNING)
print_rank0("test error", level=logging.ERROR)
print("done")