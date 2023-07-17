from src.common.registry import Registry
from src.common.example import Example
from typing import Dict, List, Generator
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


    def evaluate_examples(self, examples: List[Example], metrics_cfg: Dict=None) -> Dict:
        """ Perform evaluation with given examples of predictions.
          Args:
            @examples: a list of predicted examples instanced by `Example` class.
          Return:
            the evaluation scores calculated by each task.
        """
        pass


    def save_canonical_results(self, examples: List[Example], save_dir: str):
        """ Format and save the examples of predictions to json for future comparison to golden files.
          Args:
            @examples: a list of predicted examples.
            @save_dir: a directory string for saving the result.
        """
        pass


    def evaluate_files(self, eval_tasks: List[str], eval_files: Dict[str], metrics_cfg: Dict=None) -> Dict:
        """ Perform evaluation with given prediction files saved with canonical format.
          Args:
            @eval_tasks: a list of task names for evaluation.
            @eval_files: a list of prediction files with canonical format.
          Return:
            the evaluation scores calculated by each task.
        """
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_tasks', type=str, nargs='+', default=['VQAv2'],
                        help='Specify the tasks for evaluation, where the supported are [VQAv2, Visual7W]')
    parser.add_argument('--eval_files', type=str, nargs='+', default=[None],
                        help='Provide the prediction files saved with canonical format. The count must equal to the tasks.')
    args = parser.parse_args()

    evaluator = Evaluator()
    scores = evaluator.evaluate_files(args.eval_tasks, args.eval_files)
    print(scores)