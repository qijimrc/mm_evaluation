import os
import re
import importlib
import pkgutil

from sat.helpers import print_rank0
from braceexpand import braceexpand

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        if loader.path in package.__path__:
            full_name = package.__name__ + '.' + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
    return results

def is_chinese(text):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

def find_all_files(path, suffix=".tar"):
    pathes = path.split(',')
    target_files = []
    for p in pathes:
        if '*' in p:
            include_dirs, n = p.split('*')
            repeat_nums = int(n)
        else:
            include_dirs = p
            repeat_nums = 1
        if include_dirs.endswith(suffix): # path/to/name-{000000..000024}.tar
            target_files.extend(list(braceexpand(include_dirs)) * repeat_nums)
        else: # path/to/dataset_name
            for cur_dir, _, files in os.walk(include_dirs, followlinks=True):
                for f in files:
                    if f.endswith(suffix) and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                        target_files.extend([os.path.join(cur_dir,f)]*repeat_nums)
    print_rank0(f'find {len(target_files)} files in all...')
    return target_files

def check_config(config):
    """
    Args:
        config (_type_): config.yaml
    """
    for level_name, level_value in config["tasks"].items():
        for task_name, task_value in level_value.items():
            task_cfg = {}
            for key in ["data", "data_params", "finetune_params", "eval_params"]:
                task_cfg.update(task_value[key])
                task_value.pop(key)
            for key, value in task_value.items():
                task_cfg[key] = value
            # check data
            if task_cfg.get("eval_interval", None) != 0 and \
                task_cfg.get("eval_interval", None) != None and \
                    task_cfg.get("split", [1,1,1]) == "1" and \
                    task_cfg.get("valid_data", None) is None:
                raise ValueError(f"{task_name}/{level_name} has eval_interval but no valid_data")
            if task_cfg.get("need_finetune", False) and \
                task_cfg.get("train_data", None) is None:
                raise ValueError(f"{task_name}/{level_name} has need_finetune but no train_data")
            if task_cfg.get("need_evaluate", False) and \
                task_cfg.get("test_data", None) is None:
                raise ValueError(f"{task_name}/{level_name} has need_evaluate but no test_data")
            # check params