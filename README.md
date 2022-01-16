# Power Grid Gans
This project contains multiple GANs which are used to generate power grid related data for simulations.

## Requirements
To run the code python3.9 is required. All package requirements are listed in
the *requirements.txt*

## Getting Started
It is recommended to run this code in a virtual python environment. To do this,
install *virtualenv* using pip with `pip3 install virtualenv` on Linux.
It is also recommended to use pip-tools.

To create the virtual environment use *virtualenv*:
```bash
virtualenv -p python3 power-grid-gans
# or
python3 -m venv power-grid-gans
```

Activate the virtual environment by running
```bash
source power-grid-gans/bin/activate
```

Next install the requirements from the *requirements.txt*
If you want to use pip-tools, install it first and run:
```bash
pip-sync
```
If you are using pip execute the following:
```bash
pip install -r requirements.txt
```

Deactivate virtual environment by running
```bash
deactivate
```


## Development

### Running Tests
```bash
python3 setup.py pytest
```
### Profiling
#### 1. Tensorboard
When training models we can use tensorboard to keep track of out models.
Run tensorboard with the following command:
```bash
tensorboard --logdir . 
```

### Problems
#### 1. Missing CUDA Version of  torch
Currently pip-tools cannot specify where to download torch and what specific flavor so for cuda support do this manually.
Pip command can be generated here: https://pytorch.org/get-started/locally/

Sample:
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Atari [FileNotFoundError: Could not find module ale_c.dll](https://github.com/openai/gym/issues/1726#issuecomment-550580367)
1. Uninstall gym and atari-py (If already installed):
```bash
pip uninstall atari-py
pip uninstall gym[atari]
```

2. Download VS build tools here: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
3. Run the VS build setup and select "C++ build tools" and install it.
4. Restart PC.

5. Install cmake, atari-py and gym
```bash
pip install cmake
pip install atari-py
pip install gym[atari]
```

# Misc
## Create .gif from images
```bash
ffmpeg -f image2 -framerate 10 -i img_%03d.png -loop -1 animated.gif
```

# Docs
tbd
