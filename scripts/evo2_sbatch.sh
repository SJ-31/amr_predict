#!/usr/bin/env bash
#SBATCH --job-name=evo2
#SBATCH --partition=gpu
#SBATCH --qos=gpu40g
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --mem=64G

/data/home/shannc/amr_predict/scripts/run_evo2.sh "${@}"
