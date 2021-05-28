#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=100M
#SBATCH --output=out.out
#SBATCH --partition=debug

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/timonej3/.conda/envs/diffeq/lib/
module load miniconda
source activate diffeq
srun echo "User: $USER, node: $HOSTNAME"
python curve.py

