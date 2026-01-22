# openCAT

## Installation ##

1. Clone the repository:
```shell
   git clone https://github.com/Axian12138/Click-and-Traverse.git
```

2. Create a virtual environment and install dependencies:
```shell
   export PATH=/usr/local/cuda-12.5/bin:$PATH # cuda12 is recommended
   uv sync -i https://pypi.org/simple 
```

3. Customize the `.env` file in the root directory

4. Initialize the MuJoCo environment(creat softlink for mojuco assets):
   ```shell
   source .venv/bin/activate; source .env;
   python -m cat_ppo.utils.mj_playground_init
   ```

## Usage ##

1. Train the model
args details see `train_batch.py`
```shell
   export PATH=/usr/local/cuda-12.5/bin:$PATH
   source .env; source .venv/bin/activate; 

   python train_batch.py
```


2. Evaluate the model
```shell
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name 11031413_G1LocoPF9_v4_xP2xMxK004xhighcorner --pri

python -m cat_ppo.eval.mj_onnx_play --task G1CatPri --exp_name 01202236_G1Cat_debug_xT00xempty --pri
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name 12151522_G1LocoPFR10_SlowV2OdonoiseV2_xP0xMxK00xnarrow1
```

If you change the observation space, you may need to manually convert the model checkpoint to the ONNX format:
```shell
python -m cat_ppo.eval.brax2onnx --task G1Cat --exp_name 01202236_G1Cat_debug_xT00xempty
```