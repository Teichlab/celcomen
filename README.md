# Cell Communication Energy (celcomen)
Causal generative model designed to disentangle intercellular and intracellular gene regulation with theoretical identifiability guarantees. Celcomen can generate counterfactual spatial transcriptomic samples by simulating the effect of local perturbations, such as gene activations/inhibitations or cell insertions/deletions. 

You can find out more by reading our [manuscript](https://arxiv.org/abs/2409.05804).

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

Causal Disentanglement and spatial Counterfactuals
============
To learn intracellular and extra-cellular gene regulation and then use it to simulate inflammation conuterfactuals in specific locaitons of the tissue, follow the tutorial `analysis.perturbation_newest_celcomen.ipynb`.

As explained in the tutorial, the adata object should have count data, without any prior normalization or log-transformation.

To speed up the training process on a GPU refer to the tutorial `train_using_dataloaders_gpu.ipynb`.

