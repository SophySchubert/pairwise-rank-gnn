# Pairwise ranking GNN
This repository contains the code from my master's thesis: "Graph Rankings with Pairwise GNNs".

## Installation
Works with python version 3.9.21 and torch with cuda 11.8
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