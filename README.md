# Pairwise ranking GNN
This repo contains the code from my master's thesis: "Improving pairwise ranking problems with graph neural networks".

## Installation
Works with python version 3.8.28 and torch with cuda 11.8
```bash
  pip install -r requirements.txt
```

## Usage
- choose a config from src/config OR
- create a config file and place it in the src/config folder
```bash
  python src/experiment.py src/config/<CONFIG_NAME>.yml
```

## experiments - TODO
default => 1, fc_weight => 2, fc_extra => 3
model_units=64 => 1, 32 => 2
learnrate=0.01 => 1, 0.001 => 2
seed=42 => 42, 43 => 43

| Task      | model_units | Learnrate | seed | Status  | Folder            |
|---------|--------------|-----------|------|--------|---------------|
| default   | 64          | 0.01      | 42 | &#9745; | 2025-03-16-10-26-29 |
| default   | 64          | 0.01      | 43 | &#9745; | 2025-03-16-11-44-17 |
| default   | 64          | 0.001     | 42 | &#9745; | 2025-03-16-13-03-45 |
| default   | 64          | 0.001     | 43 | &#9745; | 2025-03-16-14-10-28 |
| default   | 32          | 0.01      | 42 | &#9745; | 2025-03-16-16-01-32 |
| default   | 32          | 0.01      | 43 | &#9744; | - |
| default   | 32          | 0.001     | 42 | gerade | 2025-03-16-17-19-30 |
| default   | 32          | 0.001     | 43 | &#9744; | - |
| fc_weight | tbd         | tbd     | tbd   | &#9744; | - |
| fc_extra  | tbd         | tbd     | tbd   | &#9744; | - |

| Unchecked | Checked |
| --------- | ------- |
| &#9744;   | &#9745; |