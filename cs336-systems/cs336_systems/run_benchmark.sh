#!/bin/bash
#SBATCH --job-name=dist_comm_bench      
#SBATCH --account=ECE491B               
#SBATCH --partition=ece491b             
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=4               
#SBATCH --gres=gpu:6                    
#SBATCH --mem=32G                       
#SBATCH --time=00:30:00                 
#SBATCH --output=allreduce_results.csv


module load lang/Anaconda3
conda activate cs336_systems

srun python benchmark_distributerd_comm.py \
     --backends gloo nccl \
     --devices cpu cuda      \
     --sizes 524288 1048576 10485760 52428800 104857600 524288000 1073741824 \
     --world-sizes 2 4 6
    > allreduce_results.csv
