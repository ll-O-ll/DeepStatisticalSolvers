#!/bin/bash
#SBATCH --mail-user=ahmed.rosanally@mail.utoronto.ca # request notification by email of jobs
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source ENV/bin/activate
python ./tensorflow-test.py

# python ./main.py --data_dir=datasets/acpf_14 --learning_rate=3e-3 --minibatch_size=1000 --alpha=1e-2 --hidden_layers=2 --latent_dimension=10 --correction_updates=10 --track_validation=1000
