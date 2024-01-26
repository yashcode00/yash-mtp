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
from datasets import load_dataset, load_metric

torch.cuda.empty_cache()

## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
save_path = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Outputs"
final_path = "/nlsasfs/home/nltm-st/sujitk/yash/datasets"


## loadfing data from disk
print("Loading the data from the disk.. wait")
# # loading the autdio dataset from the directory
# loading the autdio dataset from the directory
directory  = "/nlsasfs/home/nltm-st/sujitk/yash/datasets/resampled_data_SilencedAndOneSecondData"
print(os.listdir(directory))
data = []

for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    if path in ['eng', 'tel' ,'mar' ,'odi' ,'asm' ,'guj' ,'hin' ,'tam', 'kan' ,'mal', 'ben']:
        pathHere = os.path.join(directory, path);
        count = 0;
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

# save_path = "/kaggle/working"

# Split the data into train, eval, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["language"])
eval_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=101, stratify=temp_df["language"])

# Reset the index for all dataframes
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the train, eval, and test data as CSV files
train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
eval_df.to_csv(f"{save_path}/eval.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

print("Train df is ", train_df.shape)
print("Validation df is ",eval_df.shape)
print("Test df is ",test_df.shape)


## ############### loading the data ######################
# Loading the created dataset using datasets
# !pip install -q datasets==2.14.4

data_files = {
    "train": f"{save_path}/train.csv", 
    "validation": f"{save_path}/eval.csv",
    "test": f"{save_path}/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

print(train_dataset)
# print(eval_dataset)
# print(test_dataset)

final_path= os.path.join(final_path,"saved_dataset.hf")
print("SAving the dataset to be further use at ",final_path)
dataset.save_to_disk(final_path)

################# Done saving ##############################
print("TEsting saved dataset for success....")
 
## loading from saved dataset
dataset = load_from_disk(final_path)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

print("Train: ",train_dataset)
print("Validation: ",eval_dataset)
print("Test: ",test_dataset)

## defining input and output columns
# We need to specify the input and output column
input_column = "path"
output_column = "language"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

# Preprocess
# The next step is to load a Wav2Vec2 feature extractor to process the audio signal:
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_url , cache_dir=cache_dir)

target_sampling_rate = feature_extractor.sampling_rate
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

# def preprocess_function(examples):
#     speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
#     target_list = [label_to_id(label, label_list) for label in examples[output_column]]

#     result = processor(speech_list, sampling_rate=target_sampling_rate)
#     result["labels"] = list(target_list)

#     return result
def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    inputs = feature_extractor(
        speech_list, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    inputs["labels"] = list(target_list)
    return inputs



# train_dataset = train_dataset.map(
#     preprocess_function,
#     batch_size=100,
#     batched=True,
#     num_proc=4
# )
# eval_dataset = eval_dataset.map(
#     preprocess_function,
#     batch_size=100,
#     batched=True,
#     num_proc=4
# )
    
label2id={label: i for i, label in enumerate(label_list)}
id2label={i: label for i, label in enumerate(label_list)}

### loading the processor and tokenizer contained inside it
pooling_mode = "mean"
# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    cache_dir=cache_dir,
)
setattr(config, 'pooling_mode', pooling_mode)


processor = Wav2Vec2Processor.from_pretrained(repo_url, cache_dir=cache_dir)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
is_regression = False

    
## loading the main models
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
#     num_labels=num_labels,
#     label2id={label: i for i, label in enumerate(label_list)},
#     id2label={i: label for i, label in enumerate(label_list)},
    config=config , 
    cache_dir=cache_dir,
)

print("Work done mate")