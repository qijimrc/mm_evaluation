# README for benchmarking

### environment

```
pip install -r requirements.txt
```

- Note: Some of the models require additional dependencies. Please refer to the model file for more details.
- Xcomposer requires 'flash-attention' package, which could be installed by 
```[python]
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
cd csrc 
rm -r cutlass
git clone git@github.com:NVIDIA/cutlass.git
python setup.py install
cd csrc/rotary
pip install -e .
```
- Yi requires some local files (modified from the original LLaVa repo), so we need to set ```root``` directory in the model file to the local directory ('/share/home/chengyean/evaluation/Yi', see the model file for more details)
- Qwen-VL-Plus in ```mmdoctor/api_models``` requires a DASHSCOPE_API_KEY, please contact chengyean for his personal key.

### models

- support multi-gpu loading using accelerate.device_map, please refer to ```mmdoctor/infer_models/yi.py class YiVL_34B``` for an example.