#!/bin/bash
#SBATCH --array=0-3
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=1200M
#SBATCH --output=txt/job-%A_%a.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/timonej3/.conda/envs/diffeq/lib/
module load miniconda
source activate diffeq
srun echo "User: $USER, node: $HOSTNAME"
python examples.py $SLURM_ARRAY_TASK_ID 5

