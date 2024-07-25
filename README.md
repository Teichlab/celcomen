# Cell Communication Energy (celcomen)
Causally generative model designed to indentifiably disentangle intercellular and intracellular gene regulation.


Installation
============

Prerequisites
--
Conda Environment
--
We recommend using [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a conda environment for using celcomen. You can create a python environment using the following command:

    conda create -n celcomen_env python=3.9

Then, you can activate the environment using:

    conda activate celcomen_env


- Install pytorch (This version of dis2p is tested with pytorch 2.1.2 and cuda 12, install the appropriate version of pytorch for your system.)
```
#conda create --prefix /nfs/team205/sm58/packages/celcomen_trials/pyg_env python=3.10 -y
#conda activate /nfs/team205/sm58/packages/celcomen_trials/pyg_env
pip install scanpy
pip install matplotlib
conda install jupyterlab -y

pip install ipykernel
python -m ipykernel install --user --name celcomen_package_env

pip install torch_geometric
pip install torch torchvision torchaudio
pip install torch-cluster

pip install torcheval
```
