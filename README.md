# A Unified Multimodal Benchmark

## Design

### Abilities
This benchmark evaluate the MLLMs's abilities from 3 levels:

- Corse-grained Conceptually Understanding
    
    *Know the world basically: align vision to concepts*
    
    + recognize objects with conceptual classes and identify their spatial relationships
        - Visual7W
    + recognize the positions of objects
        - **Flickr30k-Entities (Visual Grounding)**
    + counting objects with conceptual classes
        - TDIUC
    + describing the existence and activity in visual scene
        - **NoCaps (Image Captioning)**
        - **VQAv2**
        - TextVQA
    + recognize composition of objects
        - **GQA**

- Fine-grained Specifically Understanding**
    
    *Know the world specifically: align vision to entities*
    
    + align objects to world entities
        - **OK-VQA**
        
        e.g., a picture of Tsinghua is “Tsinghua”
        
    + align composition of objects to world entities
        - new dataset for solving **math problem?**
        
        e.g., formula
        
    + align relationships between objects with semantic relations
        - new dataset?
    + align visual scene with world actual events
        - from **MOCHEG**

- Associational Understanding
    
    *Know the world thoroughly: image, reasoning and planing on current visual scene*
    
    + associate objects with other similar objects that do not exist in current image
    + associate relations with other similar relations
    + associate composition of objects with other compositions
        - **kosmos-iq50 [16]**
    + associate events with other similar events
    - metaphor, joke
        - **HatefulMemes**
    + hallucination
        - new datasets?
        - knowledge hallucination from OK-VQA
        - existence hallucination from TDIUC
        - attribute hallucination from VAW
    + Interactions in Embodied Environment
        - **Language Table**

### Input/Output Format
Each example in our benchmark is conformed with a unified format:

```
{
    'task': // the task name
    'idx': // the example index
    'vis': // the vision path
    'context': // the optinal context for input
    'question': // the input question
    'answers':  // the list of language answers
}
```

## Code

For the sake of readability, some details have been omitted.

```
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
    ├── evaluator.py
    └── __init__.py

```

## Usage

We provide two types of usages for flexibility.

### Incorporating the evaluation in your code

```Python
from mmbench.common.example import Example
from mmbench.evaluation import Evaluator

model = YourModel # your model
evaluator = Evaluator()
predictions = []
for ex in evaluator.get_mixed_dataloader():
    ans = model(ex.vis, ex.context, ex.question)
    predictions.append(Example(task=ex.task, idx=ex.idx, answers=[ans]))
scores = evaluator.evaluate_examples(predictions)
print(scores)
```

### Performing the evaluation on saved results

```Python
python mmbench.evaluator --result_file 'Path/To/YourResult.json'
```