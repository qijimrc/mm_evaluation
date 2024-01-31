import re
import jsonlines
import pandas as pd

from typing import Dict
from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask

@Registry.register_task('MMBench')
class MMBenchTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'MMBenchTask'
        super().__init__(task_cfg, **kw_args)

    def tell_right(self, response, ans):
        obj = re.search(r"[ABCD]", response)
        if not obj:
            return -1
        if obj.group() == ans.strip()[0]:
            return 1
        return 0
    
    def calc_scores(self, args, result_df) -> Dict:
        all_metrics = {}

        result_df["prompt_id"] = result_df["question_id"].apply(lambda x: x.split("_")[0])
        unfollowed_nums = 0
        categorys, normal_res, circular_res = [], [], []
        detail_results = []
        for c_id, sub_df in result_df.groupby("prompt_id"):
            circular_right = 1
            for i in range(len(sub_df)):
                row = sub_df.iloc[i]
                is_right = self.tell_right(row["preds"], row["answer"])
                if is_right == -1:
                    unfollowed_nums += 1
                if i == 0:
                    categorys.append(eval(row["orig_dict"])["category"])
                    normal_res.append(int(is_right == 1))
                circular_right = int(circular_right and is_right == 1)
                detail_results.append({
                    "question_id": row["prompt_id"],
                    "question": row["question"],
                    "preds": row["preds"],
                    "answer": row["answer"],
                    "circular_idx": row["circ_idx"] 
                })
            circular_res.append(circular_right)
        # save detail results
        save_path = f"{args.save_details_result_path}-{self.mode}.jsonl"
        with jsonlines.open(save_path, "w") as fp:
            for res in detail_results:
                fp.write(res)
            self.print_info(f"Detail results are saved into {save_path}!")
            
        assert len(normal_res) == len(circular_res), f"Not same length: normal_res is {len(normal_res)} and circular_res is {len(circular_res)}."
        self.print_info(f"Unfollowed nums: {unfollowed_nums} / {len(result_df)}")
        all_metrics["total"] = {
            "normal_acc": round(sum(normal_res) / len(normal_res), 4),
            "circular_acc": round(sum(circular_res) / len(circular_res), 4),
            "sample_nums": len(normal_res)
        }

        type_df = pd.DataFrame({"categorys": categorys, "normal_res": normal_res, "circular_res": circular_res})
        for category, sub_df in type_df.groupby("categorys"):
            all_metrics[category] = {
                "normal_acc": round(sum(sub_df["normal_res"]) / len(sub_df), 4),
                "circular_acc": round(sum(sub_df["circular_res"]) / len(sub_df), 4),
                "sample_nums": len(sub_df)
            }
        
        return all_metrics
        