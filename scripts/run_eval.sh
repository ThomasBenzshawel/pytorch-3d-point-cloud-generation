#!/bin/bash

#SBATCH --job-name="3D Experiment Larger Network"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benzshawelt@msoe.edu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=practicum


SCRIPT_NAME="3D_Experiment_Larger_Network"
CONTAINER="/data/containers/msoe-pytorch-23.05-py3.sif"



PYTHON_FILE3="evaluate.py" 
SCRIPT_ARGS3="--model ORIG_STG2 --experiment ${SCRIPT_NAME} \
	--loadPath ORIG_STG2_${SCRIPT_NAME}\
	--chunkSize 32 --batchSize 32 \
	--gpu 1 --path /data/csc4801/KedzioraLab/data_3d_point_cloud_generation"

PYTHON_FILE4="evaluate_dist_overall.py"
SCRIPT_ARGS4="--model ORIG_STG2 --experiment ${SCRIPT_NAME} \
	--loadPath ORIG_STG2_${SCRIPT_NAME}\
	--chunkSize 32 --batchSize 32 \
	--gpu 1 --path /data/csc4801/KedzioraLab/data_3d_point_cloud_generation"


## SCRIPT
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun hostname; pwd; date;
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE3} ${SCRIPT_ARGS3}
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE4} ${SCRIPT_ARGS4}

echo "END: " $SCRIPT_NAME