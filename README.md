# A Unified Multimodal Benchmark

## Design

### Abilities
This benchmark evaluate the MLLMs's abilities from 3 levels:

- Corse-grained Conceptually Understanding
    
    *Know the world basically: align vision to concepts*
    
    + recognize objects with conceptual classes and identify their spatial relationships
    + recognize the positions of objects
    + counting objects with conceptual classes
    + describing the existence and activity in visual scene
    + recognize composition of objects

- Fine-grained Specifically Understanding**
    
    *Know the world specifically: align vision to entities*
    
    + align objects to world entities
    + align composition of objects to world entities
    + align relationships between objects with semantic relations
    + align visual scene with world actual events

- Associational Understanding
    
    *Know the world thoroughly: image, reasoning and planing on current visual scene*
    
    + associate objects with other similar objects that do not exist in current image
    + associate relations with other similar relations
    + associate composition of objects with other compositions
    + associate events with other similar events
    - metaphor, joke
    + hallucination
        - knowledge hallucination
        - existence hallucination
        - attribute hallucination
    + Interactions in Embodied Environment


## Code

For the sake of readability, some details have been omitted.

```python
.
├── README.md
├── data                    # data processor, raw data -> processed data
└── mmbench
    ├── common
    │   ├── example.py
    │   ├── training.py     # main training, ref sat
    │   ├── inference.py    # main testing, ref sat
    │   ├── model.py        # model interface
    │   ├── registry.py     # registry
    │   └── utils.py        # common functions
    ├── metrics             # all metrics
    │   ├── bleu
    │   └── rouge
    │   └── acc
    │   └── vqa_acc
    └── tasks               # all tasks
    │   ├── base_task.py    # main task, including most of functions in evaluating
    │   ├── level_1
    │   │   ├── VQAv2
    │   │   └── Visual7W
    │   │   └── ...
    │   ├── level_2
    │   │   └── OK-VQA
    │   └── level_3
    │       └── HalVQA
    ├── __init__.py
    ├── config.yaml         # configure data paths, params, and other
    └── evaluator.py        # main entry
```

## Usage

### Install

Install this repo from source.

```
git clone git@github.com:qijimrc/mm_evaluation.git
cd mm_evaluation & python3 setup.py install
```

### Example

```Python
import argsparse
from mmbench.evaluator import Evaluator
from mmbench.common.model import ModelInterface

parser = argparse.ArgumentParser()
parser.add_argument('--eval_tasks', type=str, nargs='+', help='Specify the tasks for evaluation')
parser.add_argument("--custom_cfg_path", type=str, help="customized eval config path")
args = parser.parse_args()

# build your ModelInterface
mt = ModelInterface(args, model, ...)
# Evaluate
evaluator = Evaluator(custom_cfg_path=args.custom_cfg_path, custom_functions={})
scores = evaluator.evaluate(args, mt, eval_tasks=args.eval_tasks)
```

### Features

1. customized params

Create a customized `yaml` config referring tasks in the `mmbench/config.yaml`. Then add the custom_cfg_path in the `args` when you build the `Evaluator`.

2. customized functions

You can customized the following functions in the `mmbench/tasks/base_task.py`

- preprocess_datab_eval
- collate_fn
- forward_step
- forward_step_eval

## Leaderboard


| **Model**                        | **level_1** |           |          |  **level_2**|           |          | **level_3** |           |          |  **AVG**      |
|:-------------------------------- | :---------- | :---------|:---------| :---------- | :---------|:---------| :---------- | :---------|:---------| :------------ |
|                                  |   *VQAv2*   |           |          |             |           |          |  *HalVQA*   |           |          |               |
| BLIP2                            |             |           |          |             |           |          |             |           |          |               |
| LLaVA                            |             |           |          |             |           |          |             |           |          |               |
| VisualGLM-6B                     |             |           |          |             |           |          |             |           |          |               |