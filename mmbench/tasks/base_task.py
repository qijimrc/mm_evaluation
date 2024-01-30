import os
import torch
import logging
import pandas as pd

from sat.helpers import print_rank0
from mmbench.common.inference import inference_main
from mmbench.common.global_vars import *
from mmbench.dataset import ItemDataset

class BaseTask(object):
    def __init__(self,
                 task_cfg,
                 custom_functions=dict(),
                 custom_dataset_functions=dict()):
        self.task_cfg = task_cfg
        self.custom_functions = custom_functions
        self.custom_dataset_functions = custom_dataset_functions
        self.dataloader_mirror = None

    def print_info(self, print_str, add_sep_lines=False, sep_char='#', level=logging.INFO):
        print_items = [self.eval_model_name]
        if self.eval_task_name:
            print_items.append(self.eval_task_name)
        print_str = f"[{'-'.join(print_items)}] {print_str}"
        if add_sep_lines:
            print_str = f"{sep_char * 80}\n{print_str}\n{sep_char * 80}"
        print_rank0(print_str, level=level)
    
    @NotImplementedError
    def calc_scores(self, args, results_total):
        pass

    def fetch_dataset_mirror(self, args):
        """save all data to pd.DataFrame for computing metric scores
        """
        self.print_info(f'[{self.mode}]: fetch data mirror begin.')
        def get_data(dataloader):
            result = []
            top_keys, meta_keys = ["datatype", "question_id"], None
            for item in dataloader:
                qa = item["json"]
                if meta_keys is None:
                    meta_keys = list(qa["metadata"].keys())
                c_res = [qa[k] for k in top_keys] + [qa["metadata"][k] for k in meta_keys]
                result.append(c_res)
            return pd.DataFrame(result, columns=top_keys + meta_keys, dtype=str)

        mirror_df = get_data(self.dataloader_mirror)
        self.print_info(f'fetch {self.mode} data mirror end.')
        return mirror_df

    def compute_metrics(self, args, model_results):
        question_ids, preds = model_results["question_ids"], model_results["preds"]
        res_df = pd.DataFrame({"question_id": question_ids, "preds": preds}, dtype=str)
        # post process
        res_df = res_df[res_df["question_id"] != "-1"]
        res_df["preds"] = res_df["preds"].apply(lambda x: x.replace(PAD_STR, ""))
        # remove duplicates
        before_res_len = len(res_df)
        res_df = res_df.drop_duplicates(subset=["question_id"])
        if before_res_len != len(res_df):
            self.print_info(f"Sample nums change after removing duplicates: {before_res_len} -> {len(res_df)}", level=logging.WARNING)
        # get mirror data
        mirror_df = self.fetch_dataset_mirror(args)
        if not args.use_debug_mode:
            assert len(res_df) == len(mirror_df), f"Sample nums not same in test: {len(res_df)} != {len(mirror_df)}"
        res_df = res_df.merge(mirror_df, on="question_id", how="inner")
        res_df.to_csv(args.save_details_result_path, index=None)
        self.print_info(f"Save detail results in {args.save_details_result_path}...")
        return self.calc_scores(args, res_df) if self.mode != "upload" else {} 

    def collate_fn(self, all_examples):
        return all_examples

    def make_dataloader(self, args, path):
        dataset = ItemDataset(args, path, custom_functions=self.custom_dataset_functions)
        self.dataloader_mirror = dataset.data
        # dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                num_workers=0, # args.num_workers
                                                pin_memory=True,
                                                collate_fn=self.collate_fn,
                                                shuffle=False)
        return dataloader
    
    def do_evaluate(self, args, model_cls) -> dict:
        self.eval_task_name = args.eval_task_name
        self.eval_model_name = args.eval_model_name
        # update task params into args
        for k,v in self.task_cfg.items():
            setattr(args, k, v)
        # run
        task_scores = {}
        for datapath in args.datapath:
            datapath, self.mode = datapath.split("###")
            datapath = datapath if os.path.exists(datapath) else \
                os.path.join(args.data_home_dir, datapath)
            dataloader = self.make_dataloader(args, datapath)
            self.print_info(f"Start {datapath}...")
            # fetch model results
            model_results = inference_main(args, dataloader, model_cls)
            if args.rank == 0:
                # compute metrics
                task_scores[f"{args.eval_task_name}-{self.mode}"] = self.compute_metrics(args, model_results)
            self.print_info(f"End {datapath}...")
        return task_scores