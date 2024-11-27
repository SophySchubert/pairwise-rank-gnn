#!/usr/bin/bash

conda create --name convertion python=3.10
conda activate convertion
pip install node2vec==0.5.0
pip install plotly==5.24.1
pip install matplotlib==3.9.2