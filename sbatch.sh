#!/bin/bash

#SBATCH --job-name="Sbatch Example"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=student@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4

SCRIPT_NAME="Rosie Job Script For 3D Point Cloud Generation"
CONTAINER="/data/containers/msoe-pytorch-23.05-py3.sif"
PYTHON_FILE="train_stg1.py"
SCRIPT_ARGS="--model ORIG_STG1 --experiment adam_trueWD \
	--endEpoch 1000 \
	--chunkSize 100 --batchSize 100 \
	--optim adam --trueWD 1e-4 --lr 5e-3 \
	--gpu 1"

## SCRIPT
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun hostname; pwd; date;
pip install tensorboardX
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE} ${SCRIPT_ARGS}
echo "END: " $SCRIPT_NAME