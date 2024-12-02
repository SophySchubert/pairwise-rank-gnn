#!/usr/bin/bash

conda create --name experiment python=3.6
conda activate experiment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install --yes --file req_exp.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

