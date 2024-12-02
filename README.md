# Pairwise Ranking GNN
This repository contains the code for the paper "Pairwise Ranking Graph Neural Networks" by [Johannes Klicpera](https://johannesklicpera.com), [Stefan Weißenberger](https://stefanweissenberger.com), and [Stephan Günnemann](https://www.in.tum.de/daml/people/guennemann/).

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