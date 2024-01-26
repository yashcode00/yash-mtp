#! /usr/bin/env bash
#SBATCH -N 1
#SBATCH --partition=nltmp
#SBATCH --gres=gpu:8
#SBATCH --job-name=main_v1_1
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=7-0:0:0  # 7 days, 0 hours, 0 minutes, and 0 seconds (you can adjust this as need

# Define the Conda environment, activate it, and define the Python script and log file
eval "$(conda shell.bash hook)" &> error.txt
# Activate the Conda environment
conda activate wave2vec

export HTTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export HTTPS_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export FTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export ALL_PROXY='http://proxy-10g.10g.siddhi.param:9090'

# Run Python script in the background and save the output to the log file
python main_v1_1.py  &> job_output_main_v1_1.log

# Print a message indicating that the job is running in the background
echo "Job is running in the background. Check $log_file for output."

# Deactivate the Conda environment (optional)
# conda deactivate####