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



directory  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/spring-labs-resampled_data_SilenceRemovedData"
directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/rest-resampled_data_SilenceRemovedData"
root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets"
newDirectory = "spring-labs-resampled_data_SilencedAndOneSecondData"

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

mapping_words = {'SPRING_INX_Assamese_R1':'asm', 'SPRING_INX_Bengali_R1':'ben', \
                 'eng':'eng', 'SPRING_INX_Gujarati_R1':'guj', 'SPRING_INX_Hindi_R1':'hin', \
                    'SPRING_INX_Kannada_R1':'kan', 'SPRING_INX_Malayalam_R1':'mal',\
                         'SPRING_INX_Marathi_R1': 'mar', 'SPRING_INX_Odia_R1':'odi', 'SPRING_INX_Tamil_R1':'tam',\
                              'tel':'tel','SPRING_INX_Punjabi_R1':'pun'}

data = []
for path in tqdm(os.listdir(directory)):
    # print(path)
    # now eplore the inner folder ,
    #  path is actually the audio language
    pathHere = os.path.join(directory, path);
    # create new path 
    NewPathHere = CreateIfNot(newDirectory, mapping_words[path]);
    count = 0;
    if not path.startswith('.') and path!='pun':
        for subFoldlers in os.listdir(pathHere):
            if subFoldlers.startswith('Audio'):
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

