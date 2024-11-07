#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=23:00:0
#SBATCH -p compute_full_node
#SBATCH --job-name mae-3-20-ss-fs-epoch8-264000
#SBATCH --output=snr_3_20_ss_fs_epoch8_output_%j.txt
#SBATCH --mail-type=FAIL


module load anaconda3
module load cuda/11.4.4
module load gcc/11.3.0
module load openmpi/4.1.4+ucx-1.11.2
export MPLCONFIGDIR=$SCRATCH/matplotlib
export HF_HOME=$SCRATCH/huggingface
export HF_DATASETS_OFFLINE=1
source ~/.bashrc
source activate myPythonEnv
#python Trainer_mist.py
torchrun --nproc_per_node=4 --nnodes=1 Trainer_mist_complete.py
