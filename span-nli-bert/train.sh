#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt

rsync -a ada:/home2/druhan/contract-nli-bert /scratch

conda activate python-3.7
python /scratch/contract-nli-bert/train.py /scratch/contract-nli-bert/data/conf_base.yml /scratch/contract-nli-bert/results

rsync -a /scratch/contract-nli-bert ada:/home2/druhan/nli-output

