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
from packaging import version
from torch import nn
from huggingface_hub import login
import wandb
from Model import *

torch.cuda.empty_cache()
disable_caching()

## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url

# loading the autdio dataset from the directory
directory = "/home/dileep/Data2/yash_mtp/MTP-2k23-24/SilencedAndOneSecondData"
directory = "/scratch/sujeetk.scee.iitmandi/yash_mtp/MTP-2k23-24/Datasets/resampled_data_SilencedAndOneSecondData"

data = []
for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    pathHere = os.path.join(directory, path);
    count = 0;
    if not path.startswith('.') and path != "pun":
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

print("Total length of the Dataset: ", len(data), "\nDataset is something like: \n")
## now lets form a dataframe from the data array
df = pd.DataFrame(data)
print(df.head())


# Filter broken and non-existed paths
print("Filter broken and non-existed paths")
print(f"Step 0: {len(df)}")

df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", axis='columns')
print(f"Step 1: {len(df)}")
df = df.sample(frac=1)
df = df.reset_index(drop=True)

print("Labels: ", df["language"].unique())
print()


## converting this custom dataset to hugging face dataset usable satandar form
save_path = "/home/dileep/Data2/yash_mtp/MTP-2k23-24/Wav2vec-codes/Outputs"
save_path = "/scratch/sujeetk.scee.iitmandi/yash_mtp/MTP-2k23-24/Wav2vec-codes/Outputs"

# Split the data into train, eval, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["language"])
eval_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=101, stratify=temp_df["language"])

df.to_csv(f"{save_path}/df.csv", index = False)
# Reset the index for all dataframes
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the train, eval, and test data as CSV files
train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
eval_df.to_csv(f"{save_path}/eval.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

print("Train dataset csv shape: ",train_df.shape)
print("Eval dataset csv shape",eval_df.shape)
print("Test dataset csv shape",test_df.shape)


## now loading these csv's
data_files = {
    "train": f"{save_path}/train.csv", 
    "validation": f"{save_path}/eval.csv",
    "test": f"{save_path}/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t" , cache_dir="/scratch/sujeetk.scee.iitmandi/yash_mtp")
dataset.save_to_disk("/scratch/sujeetk.scee.iitmandi/yash_mtp/MTP-2k23-24/Datasets/saved_dataset.hf")
