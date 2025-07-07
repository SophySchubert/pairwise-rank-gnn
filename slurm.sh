#!/bin/bash
#SBATCH --job-name=MT-att
#SBATCH --clusters=hlai
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=sophia-%J.log

source ~/.bashrc

echo "=== JOB: $(%J) ==="
echo "=== Activating conda env ==="
conda activate sophia-pyg
echo "=== Starting script ==="
srun python3 src/experiment.py src/config/default.yml
echo "=== End of job ==="