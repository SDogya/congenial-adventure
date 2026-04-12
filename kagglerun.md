!git clone https://github.com/goombalab/hnet.git   
!git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
# !git clone https://github.com/SDogya/congenial-adventure.git

!rm -rf congenial-adventure && git clone https://github.com/SDogya/congenial-adventure.git && cp -r congenial-adventure/. . && rm -rf congenial-adventure
# !git clone https://github.com/SDogya/congenial-adventure.git
# !cp  /kaggle/working/congenial-adventure/* /kaggle/working/

import os
import subprocess

# Verify GPU
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                        capture_output=True, text=True)
print('GPU:', result.stdout.strip())

# W&B API key from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb")

    secrets = UserSecretsClient()
    os.environ['WANDB_API_KEY'] = user_secrets.get_secret("wandb")
    print('W&B API key loaded from Kaggle Secrets')
except Exception as e:
    print(f'W&B secret not found ({e}), running in offline mode')
    os.environ['WANDB_MODE'] = 'offline'

WORKDIR = '/kaggle/working'
os.chdir(WORKDIR)
print('Working dir:', os.getcwd())

!uv add wrapt zarr dill einops numba
!uv sync 

!rm -rf congenial-adventure && git clone https://github.com/SDogya/congenial-adventure.git && cp -r congenial-adventure/. . && rm -rf congenial-adventure


!uv run /kaggle/working/run.py