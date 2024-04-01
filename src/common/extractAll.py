import os
import pyzipper

zip_file_path = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/testDiralisationOutput.zip'
target_directory = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets'
password = "" 


def unzip_with_pyzipper(zip_file_path, target_directory, password):
    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # Extract the file name from the file path
    file_name = os.path.basename(zip_file_path)

    # Open the zip file with pyzipper
    with pyzipper.AESZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all contents to the target directory with the provided password
        # zip_ref.extractall(target_directory, pwd=str.encode(password))
        zip_ref.extractall(target_directory, pwd=str.encode(password))

    print(f"File extracted to: {target_directory}")

if __name__ == "__main__":
    unzip_with_pyzipper(zip_file_path, target_directory, password)


