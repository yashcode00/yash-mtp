## loading important libraries
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
from pydub import AudioSegment



directory  = "/nlsasfs/home/nltm-st/sujitk/yash/datasets/resampled_data_SilenceRemovedData"
root = "/nlsasfs/home/nltm-st/sujitk/yash/datasets"
newDirectory = "resampled_data_SilencedAndOneSecondData"

## code to break into 1 second chunks
def break_audio_into_chunks(audio_path, chunk_size=1000):
    audio = AudioSegment.from_file(audio_path)
    chunk_length = chunk_size  # in milliseconds
    chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
    return chunks

# function to check and if directory does not exists create one
def CreateIfNot(parent, path):
    if path not in os.listdir(parent):
        os.mkdir(os.path.join(parent,path))
    return os.path.join(parent,path)

# loading the autdio dataset from the directory
# if newDirectory not in os.listdir(root):
# check if directory exist
newDirectory = CreateIfNot(root, newDirectory)

data = []
for path in tqdm(os.listdir(directory)):
    # print(path)
    # now eplore the inner folder ,
    #  path is actually the audio language
    pathHere = os.path.join(directory, path);
    # create new path 
    NewPathHere = CreateIfNot(newDirectory, path);
    count = 0;
    if not path.startswith('.') and path!='pun':
        for subFoldlers in os.listdir(pathHere):
            if not subFoldlers.startswith('.'):
                # print("Inside ",subFoldlers)
                pathHere2 = os.path.join(pathHere,subFoldlers);
                NewPathHere2 = CreateIfNot(NewPathHere, subFoldlers);
            for audioFile in os.listdir(pathHere2):
                if not audioFile.startswith('.'):
                    ## Now expploring all the available audio files inside 
                    ## and if not corrupted storing then in dataframe 
                    ## extracto all req info
                    # name = audioSamples.split(".")[0]
                    finalPath = os.path.join(pathHere2,audioFile)
                    # print("Processing: ",audioFile)
                    try:
                        # Try if there are some broken files
                        speech, sr = torchaudio.load(finalPath)
                        ## break the audio into chunks of 1sec and save them again to disk
                        chunks = break_audio_into_chunks(finalPath, chunk_size=1000)  # 1 second chunks
                        f = audioFile.split(".")[0]
                        for i, chunk in enumerate(chunks):
                            chunk.export(f"{NewPathHere2}/{f}_{i+1}.wav", format="wav")  # Export each chunk to a separate file
                        count = count +1;
                    except Exception as e:
                        print(str(path), e)
        print(f'Total {count} samples loaded and saved after silence removal and 1sec duration each of {path} language dataset')

