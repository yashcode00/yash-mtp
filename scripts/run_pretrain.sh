#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --partition=nltmp
#SBATCH --gres=gpu:3
#SBATCH --job-name=wave2vec2-pretraining
#SBATCH --output=/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/out.log  # Updated output path
#SBATCH --error=/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/err.log    # Updated error path
#SBATCH --time=7-0:0:0

# Define the Conda environment, activate it, and define the Python script and log file
log_dir="/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/wav2vec2/"
output_main="${log_dir}wave2vec2_pretraining.log"

eval "$(conda shell.bash hook)" &> /nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/wav2vec2/error.txt

# Activate the Conda environment
conda activate wave2vec

# Set proxy environment variables
export HTTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export HTTPS_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export FTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export ALL_PROXY='http://proxy-10g.10g.siddhi.param:9090'

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nlsasfs/home/nltm-st/sujitk/miniconda3/envs/wave2vec/lib

# Define the command-line arguments as an array
args=(
    "--train_datasets" "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/pretraining-dataset/train.csv"
    "--val_datasets" "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/pretraining-dataset/val.csv"
    "--separator" ","
    "--logging_steps" "500"
    "--saving_steps" "500"
    "--audio_column_name" "PATH"
    "--model_name_or_path" "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/wav2vec2_pretraining-2/saved_model/epoch_2"
    "--load_from_pretrained"
    "--per_device_train_batch_size" "8"
    "--per_device_eval_batch_size" "8"
    "--num_train_epochs" "50"
    "--max_train_steps" "80000"
    "--output_dir" "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/wav2vec2_pretraining"
    "--max_duration_in_seconds" "9.0"
)

# Run the Python script with the arguments
python_script="/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/wav2vec2/pretrainWave2vec2.py"
python "${python_script}" "${args[@]}" &> "$output_main"

# Deactivate the Conda environment
conda deactivate
