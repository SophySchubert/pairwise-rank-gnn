#!/usr/bin/bash

conda create --name experiment python=3.6
conda activate experiment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.6.2
pip install spektral==1.2.0
pip install matplotlib==3.3.4
pip install plotly==5.18.0
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
