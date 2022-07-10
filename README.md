# Power Grid Gans
[![Tests](https://github.com/FlorianDe/power-grid-gans/actions/workflows/main.yml/badge.svg)](https://github.com/FlorianDe/power-grid-gans/actions/workflows/main.yml)
[![](https://tokei.rs/b1/github/FlorianDe/power-grid-gans)](https://github.com/FlorianDe/power-grid-gans) 
[![](https://tokei.rs/b1/github/FlorianDe/power-grid-gans?category=files)](https://github.com/FlorianDe/power-grid-gans)

Generative Adversarial Networks (GANs) are nowadays not only used to create novel digital artworks, but can also be found in various other domains such as text generation, music, medicine, finance, smart grids, general training data generation and many more.

This work also addresses the area of smart grids by building on the knowledge gained from other areas and existing GAN extensions to create an extensible GAN framework that makes it possible to generate novel power grid simulation data, initially using weather data, for [MIDAS](https://gitlab.com/midas-mosaik/midas).

This is an experiment-based work, where the developed methods and concepts were first applied to sinusoidal synthetic data in the form of a feasibility study, and then extended to weather data of the power grid simulation of MIDAS, tested against a set of metrics, and additionally validated visually with the help of plots. 

A GAN framework was created that allows the generation of similarly distributed weather simulation data using a CGAN. 
Great attention was paid to the extensibility and versatility of the GAN framework through an elaborated software architecture.
During the development many problems of the GAN training arose and became visible.
The generated data using the developed framework currently contains an increased temporal oscillation in contrast to the training data, which could potentially be reduced by a variety of techniques based on this work.


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
      - [CLI](#cli)

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
### CLI
Help overview
```bash
python3 -m src.pggan --help
```
### Train
```bash
python3 -m src.pggan
```
### Eval
```bash
python3 -m src.pggan --mode eval
```