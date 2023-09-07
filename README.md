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

### Input Format
- Data format in web dataset
```python
{
    "__key__": , # uni key
    "jpg": , # image bytes
    "json": [{
        "datatype": str, # 数据类型，与数据加载函数对应
        "question_id": , # 问答 id，全局递增
        "metadata": {}, # 格式定义见后文
    }...] # 同一张图片的所有问答，组装成 list
}
```

## Code

For the sake of readability, some details have been omitted.

```python
.
├── README.md
├── data
└── mmbench
    ├── common
    │   ├── example.py
    │   ├── registry.py
    │   └── utils.py
    ├── metrics
    │   ├── bleu
    │   └── rouge
    │   └── vqa_acc
    └── tasks
    │   ├── base_task.py
    │   ├── level_1
    │   │   ├── VQAv2
    │   │   │   ├── download.sh
    │   │   │   ├── vqav2_card.md
    │   │   │   └── vqav2_task.py
    │   │   └── Visual7W
    │   ├── level_2
    │   │   └── OK-VQA
    │   └── level_3
    │       └── kosmos-iq50
    ├── __init__.py
    ├── config.yaml                   # configure data paths, metrics, evaluation tasks
    └── evaluator.py

```

## Usage

We provide two types of usages for flexibility.

### Install

Install this repo from source.

### Incorporating the evaluation in your code

```Python
from mmbench.common.example import Example
from mmbench.evaluation import Evaluator

model = YourModel # your model
evaluator = Evaluator()
predictions = []
for ex in evaluator.get_mixed_dataloader():
    ans = model(ex.img_path, ex.context, ex.question)
    predictions.append(Example(task=ex.task, idx=ex.idx, answers=[ans]))
scores = evaluator.evaluate_examples(predictions)
print(scores)
```

### Performing the evaluation on saved results

```Python
python mmbench.evaluator --result_file 'Path/To/YourResult.json'
```


### Leaderboard


| **Model**                        | **level_1** |           |          |  **level_2**|           |          | **level_3** |           |          |  **AVG**      |
|:-------------------------------- | :---------- | :---------|:---------| :---------- | :---------|:---------| :---------- | :---------|:---------| :------------ |
|                                  |   *VQAv2*   |           |          |             |           |          |  *HalVQA*   |           |          |               |
| BLIP2                            |             |           |          |             |           |          |             |           |          |               |
| LLaVA                            |             |           |          |             |           |          |             |           |          |               |
| VisualGLM-6B                     |             |           |          |             |           |          |             |           |          |               |