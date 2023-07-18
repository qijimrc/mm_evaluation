from src.common.registry import Registry
from src.common.example import Example
from typing import Dict, List, Generator
from pprint import pprint
import json
import argparse



class Evaluator:
    def __init__(self, task_names: List[str]=None) -> None:
        
        if task_names is None:
            task_names = Registry.list_tasks()
        self.tasks = {
            name: Registry.get_task_class(name)() for name in task_names
        }


    def get_dataloaders(self, task_names: List[str]=None) -> Dict:
        """ Get dataloaders for specified tasks.
        Args:
          @task_names: If given, provide corresponding dataloaders;
                       otherwise provide for all supported tasks.
        Return:
          A dict keyed by task names.
        """
        dataloaders = {}
        for name in task_names:
            dataloaders[name] = self.tasks.examples
        return dataloaders

    def get_mixed_dataloader(self, ) -> Generator:
        def dataloader():
            for task in self.tasks:
                for ex in task.examples:
                    yield ex
        return dataloader


    def evaluate_examples(self, res_examples: List[Example], metrics_cfg: Dict=None) -> Dict:
        """ Perform evaluation with given examples of predictions.
          Args:
            @res_examples: a list of predicted examples instanced by `Example` class.
            @metrics_cfg: specify metrics used in each task,
                          where the pair of key and value should be the string of task name and the list of metrics names respectively.
          Return:
            the evaluation scores calculated by each task.
        """
        task_examples = {}
        for ex in res_examples:
          if ex.task not in task_examples:
            task_examples[ex.task] = [ex]
          else:
            task_examples[ex.task].append(ex)
        
        task_scores = {}
        for name in task_examples:
          scores = self.tasks[name].calc_scores(task_examples[name])
          task_scores[name] = scores
        pprint(task_scores)
        return task_scores

    def save_canonical_results(self, examples: List[Example], save_js: str):
        """ Format and save the examples of predictions to json for future comparison to golden files.
          Args:
            @examples: a list of predicted examples.
            @save_js: a full path of the json file to save result.
        """
        result = [ex.to_json() for ex in examples]
        with open(save_js, 'w') as f:
            json.dump(result, f, indent=2)


    def evaluate_files(self, canonical_file: str, metrics_cfg: Dict=None) -> Dict:
        """ Perform evaluation with given prediction files saved with canonical format.
          Args:
            @canonical_file: the result file of canonical format saved through `save_canonical_results` function.
            @metrics_cfg: specify metrics used in each task,
                          where the pair of key and value should be the string of task name and the list of metrics names respectively.
          Return:
            the evaluation scores calculated by each task.
        """
        with open(canonical_file) as f:
          result = json.load(f)
        res_examples = [Example.from_json(ex) for ex in result]

        metrics_scores = self.evaluate_examples(res_examples)
        return metrics_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_tasks', type=str, nargs='+', default=['VQAv2'],
                        help='Specify the tasks for evaluation, where the supported are [VQAv2, Visual7W]')
    parser.add_argument('--result_file', type=str, nargs='+', default=[None],
                        help='Provide the prediction files saved with canonical format. The count must equal to the tasks.')
    args = parser.parse_args()

    evaluator = Evaluator()
    scores = evaluator.evaluate_files(args.result_file)
    print(scores)