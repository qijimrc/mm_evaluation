import os
import re
import importlib
import pkgutil

from mmdoctor.common.logger import log
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
    log(f'find {len(target_files)} files in all...')
    return target_files
