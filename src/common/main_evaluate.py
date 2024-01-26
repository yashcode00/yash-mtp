## loadingh necessary modules
## loading important libraries
import numpy as np
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

from transformers import AutoFeatureExtractor
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer, is_apex_available
import numpy as np
from typing import Any, Dict, Union
import torch
from packaging import version
from torch import nn
import pandas as pd
from tqdm import tqdm
import torchaudio
import os
import IPython.display as ipd
from SilenceRemover import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import classification_report


# loading the autdio dataset from the directory
directory  = "/Users/yash/Desktop/MTP-2k23-24/resampled_data_SilencedAndOneSecondData"
directory = "/home/dileep/Data2/yash_mtp/MTP-2k23-24/TTS_data_SilenceRemovedData"
# directory = "/home/dileep/Data2/yash_mtp/MTP-2k23-24/resampled_data_SilenceRemovedData"
print(os.listdir(directory))
# directory = "/kaggle/input/silencedaudiodata/SilencedAndOneSecondData"
data = []
for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    pathHere = os.path.join(directory, path);
    count = 0;
    if not path.startswith('.'):
        for subFoldlers in os.listdir(pathHere):
            if not subFoldlers.startswith('.'):
                finalPath = os.path.join(pathHere,subFoldlers);
                name = subFoldlers.split(".")[0]
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

# ## for sub nested folder structure like in sampledata not int tts
# for path in tqdm(os.listdir(directory)):
#     # now eplore the inner folder ,
#     #  path is actually the audio language
#     pathHere = os.path.join(directory, path);
#     count = 0;
#     if not path.startswith('.') and path != "pun":
#         for subFoldlers in os.listdir(pathHere):
#             if not subFoldlers.startswith('.'):
#                 pathHere2 = os.path.join(pathHere,subFoldlers);
#                 ## Now expploring all the available audio files inside 
#                 ## and if not corrupted storing then in dataframe 
#                 for audioSamples in os.listdir(pathHere2):
#                     ## extracto all req info
#                     name = audioSamples.split(".")[0]
#                     finalPath = os.path.join(pathHere2, audioSamples);
#                     try:
#                         # There are some broken files
#                         s, sr = torchaudio.load(finalPath)
#                         data.append({
#                             "name": name,
#                             "path": finalPath,
#                             "sampling_rate": sr,
#                             "language": path,
#                         });
#                         count = count +1;
#                     except Exception as e:
#                         print(str(path), e)
#                         pass
#         print(f'Total {count} samples loaded of {path} langueage dataset')



## now lets form a dataframe from the data array
df = pd.DataFrame(data)
print("Total length of the Dataset: ", len(data))
save_path = "/Users/yash/Desktop/MTP-2k23-24/Wav2vec-codes/Outputs"
save_path = "/home/dileep/Data2/yash_mtp/MTP-2k23-24/Outputs"
# save_path = "/kaggle/working"
# Split the data into train, eval, and test sets
train_df = df
# Reset the index for all dataframes
train_df = train_df.reset_index(drop=True)
# Save the train, eval, and test data as CSV files
train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
print(train_df.shape)



# Loading the created dataset using datasets
# !pip install -q datasets==2.14.4
from datasets import load_dataset, load_metric

data_files = {
    "train": f"{save_path}/train.csv", 
    # "validation": f"{save_path}/eval.csv",
    # "test": f"{save_path}/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t" , cache_dir="/home/dileep/Data2/yash_mtp")
# dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

print(train_dataset)
# print(eval_dataset)
# print(test_dataset)




from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
#         self.pooling_mode = config.pooling_mode
        self.pooling_mode = 'mean'
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy( self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward( self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None,labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def extract_hidden_states(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Extract the hidden states from the Wave2Vec2 model
        hidden_states = outputs.last_hidden_state
        return hidden_states



# We need to specify the input and output column
input_column = "path"
output_column = "language"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Device: {device}")

model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
config = AutoConfig.from_pretrained(model_name_or_path , cache_dir="/home/dileep/Data2/yash_mtp")
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path , cache_dir="/home/dileep/Data2/yash_mtp")
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path , cache_dir="/home/dileep/Data2/yash_mtp").to(device)
# config = AutoConfig.from_pretrained(model_name_or_path )
# processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
# model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
target_sampling_rate = processor.feature_extractor.sampling_rate


label2id={label: i for i, label in enumerate(label_list)},
id2label={i: label for i, label in enumerate(label_list)},
label_names = [id2label[0][i] for i in range(num_labels)]
label_names

def speech_file_to_array_fn(batch):
    try:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech_array = torch.frombuffer(RemoveSilence(batch["path"]),dtype=torch.float32)
        speech_array = resampler(speech_array).squeeze().numpy()
        batch["speech"] = speech_array
    except:
        batch["speech"] = None
    return batch

# def speech_file_to_array_fn(batch):
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
#     speech = resampler(speech_array).squeeze().numpy()
#     return speech


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

## function for predicting the audio file language by breaking it into 
## chunks of 1sec each then padding and taking hte majority voting
def predictOne(batch):
    window_size = 16000
    silencedAudio = batch['speech']
    if silencedAudio is None:
        print("Faulty!")
        batch["predicted"] = np.random.randint(1,12)
        return batch
    # print(silencedAudio)

    overlap = window_size // 2

    windows = [silencedAudio[i:i+window_size] for i in range(0, len(silencedAudio), window_size - overlap)]
    # print(windows[0])
    preds = []
    for window in windows:
        # Extract a window of audio data
        # print("Before widnow size: ",len(window))

        ## padding and makeing the attention masks
        # Pad or truncate the audio to the target length
        audio_length = len(window)
        if audio_length < window_size:
            # If the audio is shorter, pad it with a dummy value (0 in this case)
            pad_length = window_size - audio_length
            # print(torch.zeros(pad_length))
            window = torch.cat((torch.Tensor(window), torch.zeros(pad_length)), dim=0)
        elif audio_length > window_size:
            # If the audio is longer, truncate it to the target length
            window = window[:window_size]

        # print("after size: ",len(window))
        # print("padded: ",window)

        # Predict for the window
        features = processor(window, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(device)

        # Generate the attention mask
        attention_mask = torch.ones_like(input_values)  # Default all ones
        attention_mask[0][audio_length:] = 0  # Set zeros for the padded part
        attention_mask = attention_mask.to(device)  # Add batch dimension and move to device

        # torch.set_printoptions(profile="full")
        # print("Input: ",input_values)
        # print("Attention: ",attention_mask)
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            logits = model(input_values, attention_mask=attention_mask).logits

        # Get the predicted labels for the window
        # print(logits)
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        # print("--> ",pred_ids)
        preds.append(pred_ids[0])
    # print(preds)
    # print(collections.Counter(preds).most_common(1)[0][0])
    # # Perform majority voting for all windows
    overall_majority_vote = collections.Counter(preds).most_common(1)[0][0]
    # Return the overall majority prediction
    batch["predicted"] = overall_majority_vote
    return batch


## main work starts
test_dataset = train_dataset.map(speech_file_to_array_fn)
result = test_dataset.map(predict,load_from_cache_file=False)
y_true = [config.label2id[name] for name in result["language"]]
y_pred = result["predicted"]

print(y_true[:15])
print(y_pred[:15])

result = classification_report(y_true, y_pred, target_names=label_names)
print(result)

print("Saving results to text file.....")

# Additional information to include with the report
additional_info = "classification_report_metrics_"+directory.split("/")[-1]+ "_09-10-2023_aftertuning.txt"

# Save the report with additional information to a text file
#with open(additional_info, 'w') as f:
#    f.write(result)
print("Done!")
