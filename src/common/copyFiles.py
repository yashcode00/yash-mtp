import os
import shutil

# Set the root directory and the new directory
root_directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge"
source_directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/combined-resampled_data_SilenceAndTwoSecond"
new_directory = "displace-finetuneTwoSecondSilencedData"
max_limit = 32702  # Maximum number of files to copy

# Function to create a directory if it doesn't exist
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created new directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Create the new directory if it doesn't exist
create_directory_if_not_exists(os.path.join(root_directory, new_directory))

# Source directory containing language folders
source_directory_path = source_directory
# Destination directory for the "not-eng" folder
not_eng_directory = os.path.join(root_directory, new_directory, 'not-eng')
# Create the "not-eng" folder if it doesn't exist
create_directory_if_not_exists(not_eng_directory)

# Iterate over the folders in the source directory
for lang_folder in os.listdir(source_directory_path):
    # Skip the English folder
    if lang_folder == 'eng':
        continue
    
    # Source directory for the current language
    lang_source_directory = os.path.join(source_directory_path, lang_folder)

    # Count of copied files for the current language
    lang_copied_count = 0
    
    # Iterate over the files in the language folder
    for file_name in os.listdir(lang_source_directory):
        # Check if the file is a .wav file
        if file_name.endswith('.wav'):
            # Source file path
            source_file_path = os.path.join(lang_source_directory, file_name)
            # Destination file path in the "not-eng" folder
            dest_file_path = os.path.join(not_eng_directory, file_name)
            # Copy the file to the "not-eng" folder
            shutil.copy(source_file_path, dest_file_path)
            # print(f"Copied file: {source_file_path} to {dest_file_path}")
            lang_copied_count += 1
            # Check if the maximum limit has been reached
            if lang_copied_count >= max_limit:
                break
    # Print the count of copied files for the current language
    print(f"Total {lang_copied_count} files copied for the language: {lang_folder}")
