#!/usr/bin/env bash
#SBATCH -N 4
# SBATCH --ntasks-per-node=48
#SBATCH --partition=standard
#SBATCH --job-name=evaluate-LD
#SBATCH --output=/home/sujeetk.scee.iitmandi/yash-mtp/logs/out.log  # Updated output path
#SBATCH --error=/home/sujeetk.scee.iitmandi/yash-mtp/logs/err.log    # Updated error path
#SBATCH --time=2-0:0:0  # 2 days, 0 hours, 0 minutes, and 0 seconds (you can adjust this as needed)

# Define the Conda environment, activate it, and define the Python script and log file
log_dir="/home/sujeetk.scee.iitmandi/yash-mtp/logs/wav2vec2/"
output_main="${log_dir}evaluate-LD.log"

# Activate the Conda environment
conda activate wave2vec

# Run Python script in the background and save the output to the log file
python /home/sujeetk.scee.iitmandi/yash-mtp/src/evaluate/languageDiarizer.py &> "$output_main" &

# Save the background job's process ID (PID)
bg_pid=$!

# Print a message indicating that the job is running in the background
echo "Job is running in the background with PID $bg_pid. Check job_output_main_v1_1.log for output."

# Deactivate the Conda environment (if not done already)
conda deactivate
