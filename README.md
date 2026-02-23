# Cell Communication Energy (celcomen)


[![GitHub stars](https://img.shields.io/github/stars/stathismegas/celcomen?style=social)](https://github.com/stathismegas/celcomen/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/stathismegas/celcomen?style=social)](https://github.com/stathismegas/celcomen/network/members)
[![Documentation Status](https://readthedocs.org/projects/celcomen/badge/?version=latest)](https://celcomen.readthedocs.io/en/latest/?badge=latest)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/celcomen?label=downloads)](https://pypi.org/project/celcomen/)

[![GitHub](https://img.shields.io/badge/GitHub-celcomen-181717?logo=github)](https://github.com/stathismegas/celcomen)

Causal generative model designed to disentangle intercellular and intracellular gene regulation with theoretical identifiability guarantees. Celcomen can generate counterfactual spatial transcriptomic samples by simulating the effect of local perturbations, such as gene activations/inhibitations or cell insertions/deletions. 

You can find out more by reading our [ICLR publication](https://openreview.net/forum?id=Tqdsruwyac) or [manuscript](https://arxiv.org/abs/2409.05804).    

<p align="center">
  <img src="images/disentangling graphs and gene colocalization-2.png" width="750">
</p>

Installation
============
Conda Environment
--
We recommend using [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a conda environment for using celcomen. You can create a python environment using the following command:

    conda create -n celcomen_env python=3.9

Then, you can activate the environment using:

    conda activate celcomen_env

Install celcomen
--
Then install
```
pip install git+https://github.com/stathismegas/celcomen
```

Causal Disentanglement and Spatial Counterfactuals
============
To learn intracellular and extra-cellular gene regulation and then use it to simulate inflammation conuterfactuals in specific locaitons of the tissue, follow the tutorial `analysis.spatial_KO.xenium_human_glioblastoma_gpu.ipynb`.

As explained in the tutorial, the adata object should have count data, without any prior normalization or log-transformation.

More details about the documentation can be found on [Read the Docs](https://celcomen.readthedocs.io/en/latest/).

To reproduce our results from our paper and manuscript refer to our [reproducibility repo](https://github.com/stathismegas/celcomen_reproducibility).
