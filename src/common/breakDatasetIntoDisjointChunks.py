## loading important libraries
from sklearn.model_selection import train_test_split
import os

from transformers import AutoFeatureExtractor
from datasets import load_dataset, Audio,  concatenate_datasets, Dataset, load_from_disk
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

import numpy as np
from typing import Any, Dict, Union
import torch
import os
from SilenceRemover import *

import torch
import torchaudio

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd


## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
final_path= os.path.join("/nlsasfs/home/nltm-st/sujitk/yash/datasets","saved_dataset.hf")

## loading from saved dataset
dataset = load_from_disk(final_path)
print("DAtasets loaded succesfully!!")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

print("Train: ",train_dataset)
print("Validation: ",eval_dataset)
print("Test: ",test_dataset)

print("Coombining ALL!")
df = concatenate_datasets([train_dataset, eval_dataset, test_dataset])
print("Concatenated: ",df)

### now splitting into disjoint chunks
# Split your combined dataset into chunks
chunk_size = 2000  # Example chunk size
chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

print("Total Number of chunks: ", len(chunks))
# Save each chunk as a separate dataset
for i, chunk in enumerate(chunks):
    chunk_dataset = Dataset.from_dict(chunk)
    chunk_dataset.save_to_disk(f"/nlsasfs/home/nltm-st/sujitk/yash/datasets/disjointChunks/chunk_{i}")
