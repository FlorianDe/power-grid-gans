# Power Grid Gans
![test workflow](https://github.com/FlorianDe/power-grid-gans/.github/workflows/main.yml/badge.svg)
[![](https://tokei.rs/b1/github/FlorianDe/power-grid-gans)](https://github.com/FlorianDe/power-grid-gans) 
[![](https://tokei.rs/b1/github/FlorianDe/power-grid-gans?category=files)](https://github.com/FlorianDe/power-grid-gans)

This project contains multiple GANs which are used to generate power grid related data for simulations.

## Table of Contents
  - [Requirements](#requirements)
  - [Getting Started](#getting-started)
  - [Development](#development)
    - [Running Tests](#running-tests)
    - [Profiling](#profiling)
    - [Problems](#problems)
  - [Misc](#misc)
    - [Create .gif from images](#create-gif-from-images)
  - [Docs](#docs)
    - [GAN Framework](#gan-framework)
      - [Dataflow](#data-flow)
      - [Architecture](#architecture)
  - [Experiments](#experiments)

## Requirements
To run the code python3.9 is required. All package requirements are listed in the *requirements.txt*

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
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Maybe you also need to uninstall pytorch first and purge the cache manually:

```
pip uninstall torch
pip cache purge
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

## Misc
### Create .gif from images
```bash
ffmpeg -f image2 -framerate 10 -i img_%03d.png -loop -1 animated.gif
```

## Docs
### GAN Framework
#### Dataflow
![dataflow-diagram](/.github/images/implementation/data_flow/architecture_data_flow.svg)
#### Architecture
![dataflow-diagram](/.github/images/implementation/architecture/architecture_class_diagramm.svg)
