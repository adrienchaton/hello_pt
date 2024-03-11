# Hello PyTorch!

This repository shows the basic usage of PyTorch and some related libraries for deep learning, mostly PyTorch-lightning to handle running the experiment.

Here and there, it also aims at introducing some tips and tricks to help beginners debug their codes and optimize their project.

Part of it may be used a starting point for beginners to build a template to re-use and improve in the course of their research.

Issues, contributions (and stars ;)) are welcome!


# Getting started

This is one of the many ways to setup the project.
For managing the python environments, I use mamba and install via https://github.com/conda-forge/miniforge.
In this case, it is tested on a Linux server with NVIDIA GPUs.
Please, adjust when setting up without GPU, on Windows or for Apple Silicon ...

```sh
git clone https://github.com/adrienchaton/hello_pt.git
cd hello_pt
conda update -n base -c conda-forge mamba conda
conda create -n hello_pt python=3.9
conda activate hello_pt
mamba env update -n hello_pt -f dependencies.yml
```

If all went well, you should be able to use PyTorch and the GPUs, e.g.

```sh
python
import torch
assert torch.cuda.is_available()  # should return True
```

And you should be able to import from our own package, e.g.

```sh
python
from hello_pt import *
# otherwise, you can run "pip install -e ." which will execute "setup.py" to install the package in your current env.
```
