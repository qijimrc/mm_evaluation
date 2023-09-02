import os
import re
from sat.helpers import print_rank0
from braceexpand import braceexpand


def is_chinese(text):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

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
                    if f.endswith('.tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                        tar_files.extend([os.path.join(cur_dir,f)]*repeat_nums)
    print_rank0(f'find {len(tar_files)} tars in all...')
    return tar_files