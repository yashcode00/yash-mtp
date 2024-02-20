import os
from pydub import AudioSegment

# Function to break audio into chunks
def break_audio_into_chunks(audio_path, chunk_size=1000):
    audio = AudioSegment.from_file(audio_path)
    chunk_length = chunk_size  # in milliseconds
    chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
    return chunks

# Input directory
input_dirs = ["/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_unsupervised/AUDIO_unsupervised",\
        "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_unsupervised_part2/AUDIO_unsupervised"
    ]
root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge"
newDirectory = "pretrain-dataset-unsup"

# Output directory for saving chunks
output_directory = os.path.join(root,newDirectory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through audio files in the input directory
for input_directory in input_dirs:
    for audio_file in os.listdir(input_directory):
        if audio_file.endswith(".wav"):  # Adjust file extension if necessary
            input_audio_path = os.path.join(input_directory, audio_file)
            chunks = break_audio_into_chunks(input_audio_path, chunk_size=9000)  # 8 second chunks
            file_name = os.path.splitext(audio_file)[0]  # Remove file extension
            for i, chunk in enumerate(chunks):
                output_file_path = os.path.join(output_directory, f"{file_name}_{i+1}.wav")
                chunk.export(output_file_path, format="wav")  # Export each chunk to a separate file
