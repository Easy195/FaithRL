# SPPO: Self-Play Preference Optimization for Language Model Alignment





## Reproduce the Experiment


```
cd verl
python3 -m uv pip install -e ".[sglang]"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash recipe/sppo/run_qwen2.5-7b_rm.sh
```

Note that the installation would occasionally fail to install flash-attn. If this happens, you can install it manually by running:

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## Acknowledgement

We sincerely thank the contribution and guidance from:



