#!/bin/bash
#SBATCH --job-name=cuda_cell
#SBATCH --mail-user=ksek00@zedat.fu-berlin.de   
#SBATCH --mail-type=end
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --qos=standard  

#SBATCH -o cuda.%j.out

set -e

module load CUDA/12.1.1

cd $HOME

echo "=== JOB INFO ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"
echo "================"


cd /home/ksek00/phasefield_test/


echo "=== COMPILING ==="
nvcc --std=c++17 unity.cu -lcuda -lcufft -lcublas -lstdc++fs -O3 -o cell_unity



echo "=== RUNNING ==="

./cell_unity "unity" 1 1 5 2 7 0.7 
folder1="out_debug_unity_1_1_5_2_7_0.7"


echo "Compressing folder"
tar -czf "${folder1}.tar.gz" "$folder1"


echo "=== DONE ==="
echo "Files should be in folders"


