#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## loading important libraries
import sys
import math
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging 
import shutil
import torchaudio
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoConfig, Wav2Vec2Processor
import numpy as np
from typing import Any, Dict, Union
import torch
from packaging import version
import pandas as pd
from tqdm import tqdm
import torchaudio
import os
import IPython.display as ipd
from SilenceRemover import *
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
import IPython.display as ipd
import numpy as np
import pandas as pd
import multiprocess.context as ctx
ctx._force_start_method('spawn')
from Model import *

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
def print_gpu_info():
    print("-"*20)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_capability = torch.cuda.get_device_capability(current_device)
        gpu_info = f"Number of GPUs: {device_count}\nCurrent GPU: {current_device}\nGPU Name: {device_name}\nGPU Compute Capability: {device_capability}"
        print(gpu_info)
        for i in range(device_count):
            print(f"GPU {i} Memory Usage:")
            print(torch.cuda.memory_summary(i))
    else:
        print("No GPU available.")
    print("-"*20)

print_gpu_info()

torch.multiprocessing.set_start_method('spawn')# good solution !!!!

##################################################################################################
## Important Intializations
##################################################################################################

isOneSecond = False
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-2sec-300M-saved-model-20240214_193154/pthFiles/model_epoch_0"
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
final_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-saved-dataset.hf"
hiddenFeaturesPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/tdnn"
hiddenFeatures_givenName = "displace-2sec-hiddenFeatures.hf"
givenName = "displace-2sec-HiddenFeatures_full_fast"
frames = 49 if isOneSecond else 99
cols = np.arange(0,1024,1)
chunk_size = 16000 if isOneSecond else 32000
processed_dataset_givenName = "displace-2sec-processed.hf"
processed_dataset_path  = os.path.join(hiddenFeaturesPath,processed_dataset_givenName)

# We need to specify the input and output column
input_column = "path"
output_column = "language"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

config = AutoConfig.from_pretrained(repo_url)
processor = Wav2Vec2Processor.from_pretrained(repo_url)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_url , cache_dir=cache_dir)
target_sampling_rate = processor.feature_extractor.sampling_rate
processor.feature_extractor.return_attention_mask = True ## to return the attention masks

# we need to distinguish the unique labels in our SER dataset
label_list = ['eng', 'not-eng']
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print("target sampling rate: ", target_sampling_rate)
label2id={'eng': 0, 'not-eng': 1}
id2label={0: 'eng', 1: 'not-eng'}
print(f"label2id mapping: {label2id}")
print(f"id2label mapping: {id2label}")


##################################################################################################
##################################################################################################

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    inputs = feature_extractor(
        speech_list, sampling_rate=feature_extractor.sampling_rate, max_length=chunk_size, truncation=True
    )
    inputs["labels"] = list(target_list)
    return inputs

## function to store the hidden feature representation from the last layer of wave2vec2
def hiddenFeatures(batch):
    silencedAndOneSecondAudio = batch['input_values']
    features = processor(silencedAndOneSecondAudio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    with torch.no_grad():
        # Pass attention_mask to the model to prevent attending to padded values
        hidden_features = model.extract_hidden_states(input_values)
        # print(f"Size: {hidden_features.shape}") Size: torch.Size([1024, 49, 1024]) for 1 sec
        # print(f"Size: {hidden_features.shape}") Size: torch.Size([512, 99, 1024]) for 2 sec
    batch["hidden_features"] = hidden_features
    # hFeatures = np.array(batch["hidden_features"]).reshape(1024,-1)
    for lang,name, hFeature in zip(batch['language'], batch['name'], batch['hidden_features']):
        tPath = os.path.join(targetDirectory,lang)
        fileName = lang + "_" + name.split(".")[0] + ".csv"
        if hFeature.shape[1] == 1024:
            df = pd.DataFrame(hFeature.cpu(),columns=cols)
            df.to_csv(os.path.join(tPath,fileName), encoding="utf-8", index=False)
    return batch


# function to check and if directory does not exists create one
def CreateIfNot(parent, path):
    if path not in os.listdir(parent):
        os.mkdir(os.path.join(parent,path))
    return os.path.join(parent,path)


### making new directory to store these hidden features
torch.set_num_threads(1)  ## imp

# print(train_dataset)
if __name__ == "__main__": 
    targetDirectory = os.path.join(hiddenFeaturesPath,givenName)
    if os.path.exists(targetDirectory) and os.path.isdir(targetDirectory):
        shutil.rmtree(targetDirectory)

     ### Creating folder for saving hidden featrues as csv files
    if givenName not in os.listdir(hiddenFeaturesPath):
        os.makedirs(targetDirectory)
        ##  first lets crewat all the subdirectories
        for lang in label_list:
            new_path = CreateIfNot(targetDirectory, lang)
    else:
        print(f"{os.path.join(hiddenFeaturesPath,givenName)}: Folder already exists!")
    print(f"A classification problem with {num_labels} classes: {label_list}")
    if os.path.exists(processed_dataset_path):
        logging.info("Filtered and processed dataset detected, skipping preprocessing.")
        logging.info(f"Loading from {processed_dataset_path}")
        train_dataset = load_from_disk(processed_dataset_path)
        print(f"Dataset: {train_dataset}")
        print(f"Length of each input_values is : {len(train_dataset[0]['input_values'])}")
    else:
        ## loading the dataset from saved dataset
        dataset = load_from_disk(final_path)
        print("Datasets loaded succesfully!!")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        print("Final/Train: ",train_dataset)
        print("Validation: ",eval_dataset)

        train_dataset = concatenate_datasets([train_dataset ,eval_dataset])
        train_dataset = train_dataset.map(
                            preprocess_function,
                            batch_size=1024,
                            batched=True,
                            num_proc=400,
                            # keep_in_memory=True
                            load_from_cache_file=True
        )

        # ## filtering the dataset
        # new_dataset  = []
        # print("Filtering Dataset....")
        # for row in tqdm(train_dataset):
        #     l = len(row['input_values'])
        #     if l < 900:
        #         print("skipping length too short: ",l)
        #         continue 
        #     new_dataset.append(row)

        # train_dataset =  Dataset.from_list(new_dataset)
        ## filtering the dataset
        print("Filtering Dataset....")
        train_dataset = train_dataset.filter(lambda example: len(example['input_values']) >= 900)


        print("Saving Processed dataset... ")
        try:
            train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,processed_dataset_givenName))
            logging.info(f"Saved proceesed dataset to {os.path.join(hiddenFeaturesPath,processed_dataset_givenName)}")
        except:
            logging.error("Unable to save to the disk: ", Exception)

    train_dataset = train_dataset.map(
        hiddenFeatures,
        batch_size=512,
        batched=True,
        # num_proc=400,
        # keep_in_memory=True
        # load_from_cache_file=True
    )

    print("Saving hiddenFeatures dataset... ")
    try:
        train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,hiddenFeatures_givenName))
        print("Saved hiddenFeatures to the directory: ",os.path.join(hiddenFeaturesPath,hiddenFeatures_givenName))
    except:
        logging.error("Unable to save to the disk: ", Exception)

    logging.info("Hidden Features extracted succesfully.")

    



