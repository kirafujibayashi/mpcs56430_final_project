#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --job-name=O=image_array_parallel                # Job name
#SBATCH --mail-type=ALL                              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kirafujibayashi@uchicago.edu     # Where to send mail
#SBATCH --ntasks=1                                   # Run on a single CPU
#SBATCH --mem=1gb                                    # Job memory request
#SBATCH --time=00:15:00                              # Time limit hrs:min:sec
#SBATCH --output=%j.%a.image_metrics_job.out
#SBATCH --error=%j.%a.image_metrics_job.err
#SBATCH --array=1

# Define the array suffix based on the SLURM_ARRAY_TASK_ID
# SUFFIX=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Run the Python analysis script with the suffix as an argument
python3 final_project.py 
