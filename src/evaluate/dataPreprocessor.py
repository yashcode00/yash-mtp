# This code is meant to Convert the data in out hands into 
# Silence removed Data

## loading important libraries
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from tqdm import tqdm
import torchaudio
import os
import IPython.display as ipd
from SilenceRemover import *


# function to check and if directory does not exists create one
def CreateIfNot(parent, path):
    if path not in os.listdir(parent):
        os.mkdir(os.path.join(parent,path))
    return os.path.join(parent,path)

# loading the autdio dataset from the directory
directory  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised"
root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge"
# FolderName
folderName = "Displace2024_dev_audio_supervised_SilenceRemovedData"
# check if directory exist
newDirectory = CreateIfNot(root, folderName)
# newDirectory = os.path.join(root, folderName)
data = []
for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    pathHere = os.path.join(directory, path);
    # create new path 
    NewPathHere = newDirectory;
    count = 0;
    if path.startswith('A'):
        for subFoldlers in os.listdir(pathHere):
            pathHere2 = os.path.join(pathHere,subFoldlers);
            NewPathHere2 = NewPathHere
            ## Now expploring all the available audio files inside 
            ## and if not corrupted storing then in dataframe 
            for audioSamples in os.listdir(pathHere2):
                ## extracto all req info
                # name = audioSamples.split(".")[0]
                finalPath = os.path.join(pathHere2, audioSamples);
                AudioPath = os.path.join(NewPathHere2, audioSamples);
                try:
                    # Try if there are some broken files
                    speech, sr = torchaudio.load(finalPath)
                    silencedAudio = RemoveSilence(finalPath, AudioPath)
                    count = count +1;
                except Exception as e:
                    print(str(path), e)
                    pass
        print(f'Total {count} samples loaded and saved after silence removal of {path} language dataset')


## below code works if there is only one level of nesting 

# for path in tqdm(os.listdir(directory)):
#     # now eplore the inner folder ,
#     #  path is actually the audio language
#     pathHere = os.path.join(directory, path);
#     # create new path 
#     NewPathHere = CreateIfNot(newDirectory, path);
#     count = 0;
#     if not path.startswith('.'):
#         for audioFile in os.listdir(pathHere):
#             if not audioFile.startswith('.'):
#                 ## Now expploring all the available audio files inside 
#                 ## and if not corrupted storing then in dataframe 
#                 ## extracto all req info
#                 # name = audioSamples.split(".")[0]
#                 finalPath = os.path.join(pathHere,audioFile)
#                 destinationPath = os.path.join(NewPathHere,audioFile)
#                 try:
#                     # Try if there are some broken files
#                     speech, sr = torchaudio.load(finalPath)
#                     silencedAudio = RemoveSilence(finalPath, destinationPath)
#                     count = count +1;
#                 except Exception as e:
#                     print(str(path), e)
#         print(f'Total {count} samples loaded and saved after silence removal of {path} langueage dataset')



