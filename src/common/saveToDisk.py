## author @Yash Sharma, B20241
## imporying all the neccesary modules
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

from datasets import load_dataset, Audio
from dataclasses import dataclass
from datasets import load_dataset, load_metric, load_from_disk, disable_caching
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
# is_apex_available
import numpy as np
from typing import Any, Dict, Union, Tuple
import torch
from torch import nn
from huggingface_hub import login
import wandb
from Model import *
from datasets import load_dataset, load_metric

torch.cuda.empty_cache()

## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
final_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2"


## loadfing data from disk
print("Loading the data from the disk.. wait")
# # loading the autdio dataset from the directory
# loading the autdio dataset from the directory
directory  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/resampled_data_SilencedAndOneSecondData"
print(os.listdir(directory))
data = []

for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    if path in ['eng', 'tel' ,'mar' ,'odi' ,'asm' ,'guj' ,'hin' ,'tam', 'kan' ,'mal', 'ben']:
        pathHere = os.path.join(directory, path);
        count = 0
        if not path.startswith('.'):
            for subFoldlers in os.listdir(pathHere):
                if not subFoldlers.startswith('.'):
                    pathHere2 = os.path.join(pathHere,subFoldlers);
                    ## Now expploring all the available audio files inside 
                    ## and if not corrupted storing then in dataframe 
                    for audioSamples in os.listdir(pathHere2):
                        ## extracto all req info
                        name = audioSamples.split(".")[0]
                        finalPath = os.path.join(pathHere2, audioSamples);
                        try:
                            # There are some broken files
                            s, sr = torchaudio.load(finalPath)
                            ## dummy path
                            data.append({
                                "name": name,
                                "path": finalPath,
                                "sampling_rate": sr,
                                "language": path,
                            });
                            count = count +1;
                        except Exception as e:
                            print(str(path), e)
                            pass
            print(f'Total {count} samples loaded of {path} langueage dataset')

## now lets form a dataframe from the data array
df = pd.DataFrame(data)
print("Total length of the Dataset: ", len(data))
print(df.head())
print(df.iloc[0])

## ecpplore dataset stats
print("Labels: ", df["language"].unique())
print()
df.groupby("language").count()[["path"]]

# cache_dir = "/kaggle/working"

# Split the data into train, eval, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["language"])
eval_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=101, stratify=temp_df["language"])

# Reset the index for all dataframes
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the train, eval, and test data as CSV files
train_df.to_csv(f"{cache_dir}/train.csv", sep="\t", encoding="utf-8", index=False)
eval_df.to_csv(f"{cache_dir}/eval.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{cache_dir}/test.csv", sep="\t", encoding="utf-8", index=False)

print("Train df is ", train_df.shape)
print("Validation df is ",eval_df.shape)
print("Test df is ",test_df.shape)


## ############### loading the data ######################
# Loading the created dataset using datasets
# !pip install -q datasets==2.14.4

data_files = {
    "train": f"{cache_dir}/train.csv", 
    "validation": f"{cache_dir}/eval.csv",
    "test": f"{cache_dir}/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

print(train_dataset)
# print(eval_dataset)
# print(test_dataset)

final_path= os.path.join(final_path,"saved_dataset.hf")
print("Saving the dataset to be further use at ",final_path)
dataset.save_to_disk(final_path)

################# Done saving ##############################
print("Work done mate")