# Pairwise ranking GNN
This repository contains the code from my master thesis: "Graph Rankings with Pairwise GNNs"

## Installation
Works with python version 3.9.21 and torch with cuda 11.8\
**Due to a change in the torch load api since version 2.5 you probably need to add "weights_only=False" to OGB dataset loader!**
```bash
  pip install -r requirements.txt
```

## Usage
Create or choose a config from src/config and run the experiment with:
(All currently available possibilities are listed in default.yml)
```bash
  python src/experiment.py src/config/<CONFIG_NAME>.yml
```
To restart a previously stopped experiment, do:
```bash
  python src/experiment.py experiment/<experiment-path>/config.yml experiment/<experiment-path>/epoch<epoch-number>_state.pt
```
To test the transitivity and antisymmetry score of a model (after training), you can use the following command:
```bash
  python src/experiment.py experiment/<experiment-path>/config.yml experiment/<experiment-path>/epoch<epoch-number>_state.pt properties
```

### Tutorial on how to change ogb (version 1.3.6) library
In your conda envs there is a file \envs\<env_name>\Lib\site-packages\ogb\graphproppred\dataset_pyg.py 
in line 68 you need to change the parameter from True to False.