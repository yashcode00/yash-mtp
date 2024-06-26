#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --partition=nltmp
#SBATCH --job-name=eval2
#SBATCH --gres=gpu:5
#SBATCH --output=/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/out.log  # Updated output path
#SBATCH --error=/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/err.log    # Updated error path
#SBATCH --time=7-0:0:0  # 7 days, 0 hours, 0 minutes, and 0 seconds (you can adjust this as needed)

# Define the Conda environment, activate it, and define the Python script and log file
log_dir="/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/uvector/evaluations/"
output_main="${log_dir}whisper-displace-vad-2lang-dev-16000-0.25.log"

eval "$(conda shell.bash hook)" &> /nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/wav2vec2/error.txt

# Activate the Conda environment
conda activate wave2vec
# pip3 install -I pyannote.audio
pip install openai-whisper

# pip3 install -I torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set proxy environment variables
export HTTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export HTTPS_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export FTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export ALL_PROXY='http://proxy-10g.10g.siddhi.param:9090'

# Run Python script in the background and save the output to the log file
# python3 /nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/languageDiarizer-fast-uvector.py &> "$output_main" &
# python3 /nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/languageDiarizer-fast-uvector--withVAD-springLabs.py &> "$output_main" &
python3 /nlsasfs/home/nltm-st/sujitk/yash-mtp/src/OpenAI_whisper/languageDiarizer-fast-uvector-whisper.py &> "$output_main" &

# Save the background job's process ID (PID)
bg_pid=$!

# Print a message indicating that the job is running in the background
echo "Job is running in the background with PID $bg_pid."

# Deactivate the Conda environment (optional)
# conda deactivate  

# Wait for the background job to complete
wait $bg_pid

# Deactivate the Conda environment (if not done already)
conda deactivate
