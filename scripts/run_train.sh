#!/bin/bash
#SBATCH --job-name=run_train             # create a short name for your job
#SBATCH --output=logs/factor_index0_%j.out    # stdout file
#SBATCH --error=logs/factor_index0_%j.err     # stderr file
#SBATCH --nodes=1                             # node count
#SBATCH --ntasks=1                            # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G                            # total memory per node (increased for large data)
#SBATCH --gres=gpu:1                          # number of gpus per node
#SBATCH --partition=pli                       # job partition
#SBATCH --account=qishuo_project              # account name
#SBATCH --time=24:00:00                       # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL            # send mail for job start, end, and failure

# Load environment if needed (e.g. conda activate)
# source ~/.bashrc
# conda activate IGQA_env

# Create logs directory if it doesn't exist (though SLURM expects it to exist before submission usually)
mkdir -p logs

# Run the experiment
python experiments/train.py

