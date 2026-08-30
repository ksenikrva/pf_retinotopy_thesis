#!/bin/bash

#SBATCH --job-name=cuda_cell
#SBATCH --mail-user=ksek00@zedat.fu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --array=0-89%10
#SBATCH -o cuda.%A_%a.out

set -e

module load CUDA/12.1.1

cd $HOME

cd /home/ksek00/phasefield_test/

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" lhs_params_D-rep.csv)

IFS=',' read -r adh dif rep <<< "$line"

fmi="$adh"
sdk="$adh"
repi=5

rows=3
bundles=6
tag="ver2b_Drep5_${SLURM_ARRAY_TASK_ID}"

echo "Running task $SLURM_ARRAY_TASK_ID:"
echo "repi=$repi fmi=$fmi sdk=$sdk dif=$dif"

./ver2b "$tag" "$rows" "$bundles" "$repi" "$fmi" "$sdk" "$dif"

folder="out_debug_${tag}_${rows}_${bundles}_${repi}_${fmi}_${sdk}_${dif}"

if [ -d "$folder" ]; then
    echo "Compressing $folder"
    tar -czf "${folder}.tar.gz" "$folder"
    rm -rf "$folder"
else
    echo "WARNING: folder $folder not found"
fi

