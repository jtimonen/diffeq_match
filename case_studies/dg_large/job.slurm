#!/bin/bash
#SBATCH --array=1-4
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=1200M
#SBATCH --output=txt/scvi-%a.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/timonej3/.conda/envs/diffeq/lib/
module load miniconda
source activate diffeq
srun echo "User: $USER, node: $HOSTNAME"
python fit.py $SLURM_ARRAY_TASK_ID
