# Cell Communication Energy (celcomen)
Causal generative model designed to indentifiably disentangle intercellular and intracellular gene regulation. Celcomen can generate counterfactual spatial transcriptomic samples by simulating the effect of local gene perturbations (activations or inhibitions). 


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

To speed up the training process on a GPU refer to the tutorial `train_using_dataloaders_gpu.ipynb`.

