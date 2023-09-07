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

def get_tar_files(path):
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
        
    
urls = get_tar_files("/nxchinamobile2/shared/mmbench_datasets/processed/HalVQA/train")
dataset = MetaDistributedWebDataset(urls, process_fn, 1234, meta_names=['json'])
wds_dataloader = DataLoader(dataset, num_workers=1, batch_size=8, collate_fn=collate_fn)

image_num, sample_num = 0, 0
for item in wds_dataloader:
    print(item)
    break