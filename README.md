# Pariwise ranking GNN
This repo contains the code from my masters thesis: "Imporving pairwise ranking problems with graph neural networks".

## Installation
```bash
#for experiments
sh experiment.sh
```

```bash
#for convertion from nodes to vectors and visualtization
sh convertion.sh
```

## Usage
- choose a config from src/config OR
- create a config file and place it in the src/config folder
```bash
python .\src\experiment.py .\src\config\default.json
```