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


- Install pytorch
  
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

Install celcomen
--
Then install
```
pip install git+https://github.com/stathismegas/celcomen
```

