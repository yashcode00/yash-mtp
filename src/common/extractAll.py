import os
import zipfile

source_directory = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/eng'  
target_directory = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/rest/eng'  

def unzip_spring_files():
    global target_directory
    global source_directory

    # List all files in the source directory
    files = os.listdir(source_directory)

    # Filter files ending with ".zip"
    spring_files = [file for file in files if file.endswith('.zip')]

    if not spring_files:
        print("No ZIP files found.")
        return

    for spring_file in spring_files:
        filename = os.path.splitext(spring_file)[0]  # Remove the '.zip' extension
        file_path = os.path.join(source_directory, spring_file)
        target_path = os.path.join(target_directory, filename)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)

        print(f"Successfully extracted {spring_file} to {target_path}")

if __name__ == "__main__":
    unzip_spring_files()
