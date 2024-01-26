#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## loading important libraries
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch
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


repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Saved_Models_full/pthFiles/model_epoch_3"
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
dir = "/nlsasfs/home/nltm-st/sujitk/yash/datasets"
final_path = os.path.join(dir,"saved_dataset.hf")
hiddenFeaturesPath = dir
givenName = "HiddenFeatures_full"

# loading the dataset from saved position
## loading from saved dataset
dataset = load_from_disk(final_path)
print("DAtasets loaded succesfully!!")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

train_dataset = concatenate_datasets([train_dataset ,eval_dataset, test_dataset])

print("Train: ",train_dataset)
print("Validation: ",eval_dataset)
print("Test: ",test_dataset)

# We need to specify the input and output column
input_column = "path"
output_column = "language"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

config = AutoConfig.from_pretrained(repo_url)
processor = Wav2Vec2Processor.from_pretrained(repo_url)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_url , cache_dir=cache_dir)
target_sampling_rate = processor.feature_extractor.sampling_rate

print("target sampling rate: ", target_sampling_rate)
label2id={label: i for i, label in enumerate(label_list)},
id2label={i: label for i, label in enumerate(label_list)},
print(label2id)
print(id2label)
label_names = [id2label[0][i] for i in range(num_labels)]
label_names

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

def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
#     attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values).logits 
#         logits = model(input_values, attention_mask=attention_mask).logits 
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


## function to store the hidden feature representation from the last layer of wave2vec2
def hiddenFeatures(batch):
    silencedAndOneSecondAudio = batch['input_values']
    # Predict for the window
    features = processor(silencedAndOneSecondAudio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)

    # torch.set_printoptions(profile="full")
    # print("Input: ",input_values)
    # print("Attention: ",attention_mask)
    try:
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            hidden_features = model.extract_hidden_states(input_values)
    except Exception as err:
        print(f"Error: {err}, Input Length was: {len(silencedAndOneSecondAudio)} and features len was : {len(input_values)}")
        exit()
    # Return the overall majority prediction
    # print("Hiden features size: ",hidden_features.shape)
    batch["hidden_features"] = hidden_features[0]
    return batch


# function to check and if directory does not exists create one
def CreateIfNot(parent, path):
    if path not in os.listdir(parent):
        os.mkdir(os.path.join(parent,path))
    return os.path.join(parent,path)


### making new directory to store these hidden features
torch.set_num_threads(1)  ## imp

train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=1024,
    batched=True,
    num_proc=400,
    # keep_in_memory=True
    load_from_cache_file=True
)
print(train_dataset)

# train_dataset = train_dataset.map(
#     hiddenFeatures,
#     batch_size=1024,
#     batched=True,
#     num_proc=400,
#     # keep_in_memory=True
#     load_from_cache_file=True
# )


# try:
#     train_dataset.save_to_disk(os.path.join(dir,"train_dataset_hiddenFeatures.hf"))
#     print("Saved hiddenFeatrures to the directory")
# except:
#     print("Unable to save to the disk: ", Exception)


### finally saving all the csv files for all the languages
if givenName not in os.listdir(hiddenFeaturesPath):
    cols = np.arange(0,49,1)
    targetDirectory = CreateIfNot(hiddenFeaturesPath,givenName)
    ##  first lets crewat all the subdirectories
    for lang in label_list:
        new_path = CreateIfNot(targetDirectory, lang)
    ## now we will iterate over all the dataset and store its hidden featrues as the csv file with same name
    for rows in tqdm(train_dataset):
        l = len(rows['input_values'])
        if l < 900:
            print("skipping length too short: ",l)
            continue
        temp = hiddenFeatures(rows)['hidden_features']
        hFeatures = temp.reshape(1024,-1)
        # hFeatures = np.array(rows["hidden_features"]).reshape(1024,-1)
        lang = rows['language']
        tPath = os.path.join(targetDirectory,lang)
        fileName = lang + "_" + rows['name'].split(".")[0] + ".csv"
        if hFeatures.shape[1] != 49:
            continue
        df = pd.DataFrame(hFeatures,columns=cols)
        df.to_csv(os.path.join(tPath,fileName), encoding="utf-8", index=False)
else:
    print("Folder already exists!")
print("Done.....")



