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
virtualenv -p python3 venv
```

Activate the virtual environment by running
```bash
source venv/bin/activate
```

Next install the requirements from the *requirements.txt*
If you are using pip-tools use:
```bash
pip-sync
```

If you are using pip execute the following:
```bash
pip install -r requirements.txt
```

## Developing
When training models we can use tensorboard to keep track of out models.
Run tensorboard with the following command:
```bash
tensorboard --logdir . 
```
Currently pip-tools cannot specify where to download torch and what specific flavor so for cuda support do this manually.
Pip command can be generated here: https://pytorch.org/get-started/locally/

Sample:
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
# Docs
tbd
