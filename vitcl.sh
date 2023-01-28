#!/bin/bash
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --cpus-per-gpu=8
#SBATCH --output=vitcl.out

# load modules
#module load Anaconda3/5.3.0
#/opt/ebsofts/Anaconda3/2020.07/etc/profile.d/conda.sh

echo '###################'
echo $CUDA_VISIBLE_DEVICES

echo '###################'
nvidia-smi

echo '###################'
echo "ls"
ls

echo '###################'
echo "conda --version"

conda --version

conda env list
source activate solar_new

python --version
python /srv/beegfs/scratch/users/s/shtyk1/vit_casa/classifier/main.py 