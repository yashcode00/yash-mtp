#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## loading important libraries
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
from Model import *

torch.multiprocessing.set_start_method('spawn')# good solution !!!!


repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Saved_Models_full/pthFiles/model_epoch_3"
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
dir = "/nlsasfs/home/nltm-st/sujitk/yash/datasets"
final_path = os.path.join(dir,"saved_dataset.hf")
hiddenFeaturesPath = dir
givenName = "HiddenFeatures_full_fast"
cols = np.arange(0,49,1)
processed_dataset_path  = "/nlsasfs/home/nltm-st/sujitk/yash/datasets/processed.hf"

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


# we need to distinguish the unique labels in our SER dataset
label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'tam', 'tel']
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print("target sampling rate: ", target_sampling_rate)
label2id={'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'tam': 9, 'tel': 10}
id2label={0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'tam', 10: 'tel'}
print(label2id)
print(id2label)

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
        speech_list, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
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
    batch["hidden_features"] = hidden_features.reshape(-1,1024,49)
    # hFeatures = np.array(batch["hidden_features"]).reshape(1024,-1)
    for lang,name, hFeature in zip(batch['language'], batch['name'], batch['hidden_features']):
        tPath = os.path.join(targetDirectory,lang)
        fileName = lang + "_" + name.split(".")[0] + ".csv"
        if hFeature.shape[1] == 49:
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

    # # loading the dataset from saved position
    ## loading from saved dataset
    dataset = load_from_disk(final_path)
    print("DAtasets loaded succesfully!!")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    train_dataset = concatenate_datasets([train_dataset ,eval_dataset, test_dataset])

    print("Final/Train: ",train_dataset)
    print("Validation: ",eval_dataset)
    print("Test: ",test_dataset)

    print(f"A classification problem with {num_labels} classes: {label_list}")

    targetDirectory = os.path.join(hiddenFeaturesPath,givenName)
    if os.path.exists(targetDirectory) and os.path.isdir(targetDirectory):
        shutil.rmtree(targetDirectory)

    ### finally saving all the csv files for all the languages
    if givenName not in os.listdir(hiddenFeaturesPath):
        os.makedirs(targetDirectory)
        ##  first lets crewat all the subdirectories
        for lang in label_list:
            new_path = CreateIfNot(targetDirectory, lang)
    else:
        print("Folder already exists!")
    
    print("New dataset:\n", train_dataset)

    train_dataset = train_dataset.map(
                        preprocess_function,
                        batch_size=1024,
                        batched=True,
                        num_proc=400,
                        # keep_in_memory=True
                        load_from_cache_file=True
    )


    # ## filtering the dataset
    new_dataset  = []
    print("Filtering Dataset....")
    for row in tqdm(train_dataset):
        l = len(row['input_values'])
        if l < 900:
            print("skipping length too short: ",l)
            continue 
        new_dataset.append(row)

    train_dataset =  Dataset.from_list(new_dataset)

    print("Saving Processed dataset... ")
    try:
        train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,"processed.hf"))
        print("Saved processed to the directory")
    except:
        print("Unable to save to the disk: ", Exception)

    train_dataset = train_dataset.map(
        hiddenFeatures,
        batch_size=1024,
        batched=True,
        # num_proc=400,
        # keep_in_memory=True
        # load_from_cache_file=True
    )


    print("Saving hiddenFeatures dataset... ")
    try:
        train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,"hiddenFeatures.hf"))
        print("Saved hiddenFeatures to the directory")
    except:
        print("Unable to save to the disk: ", Exception)

    print("Done.....")

    



