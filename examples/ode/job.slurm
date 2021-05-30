#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --output=job.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/timonej3/.conda/envs/diffeq/lib/
module load miniconda
source activate diffeq
srun echo "User: $USER, node: $HOSTNAME"
python curve.py

