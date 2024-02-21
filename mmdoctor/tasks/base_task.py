import os
import torch
import jsonlines

from mmdoctor.common.inference import inference_main
from mmdoctor.common.registry import Registry
from mmdoctor.common.logger import log
from mmdoctor.dataset import ItemDataset
from mmdoctor.common.global_vars import *

class BaseTask(object):
    def __init__(self,
                 task_cfg,
                 custom_data_hooks=dict()):
        self.task_cfg = task_cfg
        self.custom_data_hooks = custom_data_hooks
        self.dataloader_mirror = None

    def calc_scores(self, args, results_data):
        ret = {}
        # split types
        for metric_name in args.metrics:
            metric_cls = Registry.get_metric_class(metric_name)
            ret[metric_name] = metric_cls.calc_scores(results_data)
        return ret

    def fetch_dataset_mirror(self):
        """save all data to pd.DataFrame for computing metric scores
        """
        log(f'fetch {self.mode} data mirror begin.')
        def get_data(dataloader):
            data = {}
            for item in dataloader:
                metadata = item["json"].pop("metadata")
                c_item = {**item["json"], **metadata}
                c_item["image_path"] = item["image_path"]
                c_item["question_id"] = str(c_item["question_id"])
                data[str(item["json"]["question_id"])] = c_item
            return data

        mirror_data = get_data(self.dataloader_mirror)
        log(f'fetch {self.mode} data mirror end.')
        return mirror_data

    def compute_metrics(self, args, model_results):
        # get mirror data
        mirror_data = self.fetch_dataset_mirror()
        # merge preds
        res_data = []
        for ques, pred in zip(model_results["question_ids"], model_results["preds"]):
            res_data.append({**mirror_data[ques], "predict": pred})
        if not args.use_debug_mode:
            assert len(res_data) == len(mirror_data), f"Sample nums not same: {len(res_data)} != {len(mirror_data)}"
        # save & compute
        with jsonlines.open(f"{args.save_details_result_path}-{self.mode}.jsonl", "w") as fp:
            for res in res_data:
                fp.write(res)
        log(f"Save detail results in {args.save_details_result_path}-{self.mode}.jsonl.")
        return self.calc_scores(args, res_data) if self.mode != "upload" else {} 

    def collate_fn(self, all_examples):
        return all_examples

    def make_dataloader(self, args, path):
        dataset = ItemDataset(args, path, custom_data_hooks=self.custom_data_hooks)
        self.dataloader_mirror = dataset.data
        # dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=self.collate_fn,
                                                shuffle=False)
        return dataloader
    
    def do_evaluate(self, args, model_cls) -> dict:
        # update task params into args
        for k,v in self.task_cfg.items():
            setattr(args, k, v)
        # run
        task_scores = {}
        for datapath in args.datapath:
            datapath, self.mode = datapath.split("###")
            datapath = datapath if os.path.exists(datapath) else \
                os.path.join(args.task_cache_dir, datapath)
            dataloader = self.make_dataloader(args, datapath)
            log(f"Start {datapath}.")
            # fetch model results
            model_results = inference_main(args, dataloader, model_cls)
            if args.rank == 0:
                # compute metrics
                task_scores[f"{self.mode}"] = self.compute_metrics(args, model_results)
            log(f"End {datapath}.")
        return task_scores