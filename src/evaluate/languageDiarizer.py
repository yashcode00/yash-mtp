#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessary modules and libraries
import sys
sys.path.append("/home/sujeetk.scee.iitmandi/yash-mtp/src/common")
from Model import *
from SilenceRemover import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchaudio
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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
import numpy as np
from typing import Any, Dict, Union, Tuple
import torch
from packaging import version
from torch import nn
from huggingface_hub import login
from torch import optim
import random
from torch.autograd import Variable
import concurrent.futures
from gaussianSmooth import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
nc = 11 # Number of language classes 
n_epoch = 200 # Number of epochs
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
path = "/Users/yash/Desktop/MTP-2k23-24"
xVectormodel_path = "/Users/yash/Desktop/MTP-2k23-24/Wav2vec-codes/model_xVector.pth"
silencedAndOneSecondAudio_size = 16000
window_size = 16000
hop_length_seconds = 0.25  # Desired hop length in seconds
audio_path = "/Users/yash/Desktop/MTP-2k23-24/Wav2vec-codes/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale/HECodemixedFemale1.wav"

### Intializing models
## for wave2vec2
model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
target_sampling_rate = processor.feature_extractor.sampling_rate

target_sampling_rate

### for x-vector
model_xVector = X_vector(IP_dim, nc)
# model.cuda()
optimizer =  optim.Adam(model_xVector.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output
#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

try:
    ## only use map location for non cuda devices
    model_xVector.load_state_dict(torch.load(xVectormodel_path, map_location=torch.device('cpu')), strict=False)
except Exception as err:
    print("Error is: ",err)
    print("No, valid/corrupted TDNN saved model found, Aborting!")

label_list  = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'tam', 'tel']
lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'tam': 9, 'tel': 10}
id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'tam', 10: 'tel'}
### defining mask for the hindi and english language
mask = np.array([0,0,1,0,1,0,0,0,0,0,0])
print("Mask for binary (Hin/Eng) classification: ",mask)


def preProcessSpeech(path):
    speech_array, sampling_rate = torchaudio.load(path)
    # speech_array = torch.frombuffer(RemoveSilence(path),dtype=torch.float32)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech_array = resampler(speech_array).squeeze().numpy()
    return speech_array

## function to store the hidden feature representation from the last layer of wave2vec2
def getHiddenFeatures(speech_array):
    silencedAndOneSecondAudio = speech_array
    # print(len(silencedAndOneSecondAudio))
    preds = []
    # Predict for the silencedAndOneSecondAudio

    audio_length = len(silencedAndOneSecondAudio)
    if audio_length < silencedAndOneSecondAudio_size:
        # If the audio is shorter, pad it with a dummy value (0 in this case)
        pad_length = silencedAndOneSecondAudio_size - audio_length
        # print(torch.zeros(pad_length))
        silencedAndOneSecondAudio = torch.cat((torch.Tensor(silencedAndOneSecondAudio), torch.zeros(pad_length)), dim=0)
    elif audio_length > silencedAndOneSecondAudio_size:
        # If the audio is longer, truncate it to the target length
        silencedAndOneSecondAudio = silencedAndOneSecondAudio[:silencedAndOneSecondAudio_size]

    # print("after size: ",len(silencedAndOneSecondAudio))
    # print("padded: ",silencedAndOneSecondAudio)

    # Predict for the silencedAndOneSecondAudio
    features = processor(silencedAndOneSecondAudio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)

    # Generate the attention mask
    attention_mask = torch.ones_like(input_values)  # Default all ones
    if audio_length > 16000:
        attention_mask[0][audio_length:] = 0  # Set zeros for the padded part
    attention_mask = attention_mask.to(device)  # Add batch dimension and move to device

    # torch.set_printoptions(profile="full")
    # print("Input: ",input_values.shape)
    # print("Attention: ",attention_mask.shape)
    try:
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            hidden_features = model_wave2vec2.extract_hidden_states(input_values, attention_mask=attention_mask)
            logits = model_wave2vec2(input_values, attention_mask=attention_mask).logits 
            print("Predictions from wave2vec2: ",id2lang[torch.argmax(logits, dim=-1).detach().cpu().numpy()[0]], end=",")
    except Exception as err:
        print(f"Error -> {err} \nSKIPPED! Input Length was: {len(silencedAndOneSecondAudio)} and features len was : {input_values.shape}")
    # Return the overall majority prediction
    # print("Hiden features size: ",hidden_features.shape)
    return hidden_features

def inputTDNN(hidden_features):
    #### Function to return data (vector) and target label of a csv (MFCC features) file
    print("Shape of hidden features: ",hidden_features.shape)
    X = hidden_features.reshape(-1,1024)
    Xdata1 = []
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a = X[i:(i+look_back1),:]  
        b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
        Xdata1.append(b)
    Xdata1 = np.array(Xdata1)    
    Xdata1 = torch.from_numpy(Xdata1).float() 
    return Xdata1

def processFrames(frame):
    ## Step 2: get the hidden featrue for the processed output
    x = getHiddenFeatures(frame)
    ## Step 3: get the output of TDNN for both eng and hindi
    XX_val = inputTDNN(x)
    XX_val = torch.unsqueeze(XX_val, 1)
    X_val = np.swapaxes(XX_val, 0, 1)
    X_val = Variable(X_val, requires_grad=False)
    model_xVector.eval()  # Set the model to evaluation mode
    val_lang_op =model_xVector.forward(X_val)
    val_lang_op = val_lang_op.detach().cpu().numpy()[0]
    val_lang_op = np.array([val_lang_op[2], val_lang_op[4]])
    print("Before Softmax",val_lang_op)
    # y = dic[np.argmax(val_lang_op)]
    ## apply softmax
    val_lang_op = np.exp(val_lang_op)/np.sum(np.exp(val_lang_op))
    print("After softmax: ",val_lang_op)
    # print(f"Predicted language for this window is: ",y)
    ## Step 4: mask the output for all languauge except eng and hindi
    # dummy.append(y)
    # print(val_lang_op)
    # S0.append(val_lang_op[0])
    # S1.append(val_lang_op[1])
    return val_lang_op

def pipeline(path):
    ## Step 1: preprocees the audio by removing silence
    x = preProcessSpeech(path)
    ## Now, we will just bbreak this audio into multiple overlapping windows 
    # Calculate the hop size in samples
    hop_size = int(hop_length_seconds * target_sampling_rate)  # Adjust 'sample_rate' as needed

    # Generate overlapping frames
    frames = [x[i:i+window_size] for i in range(0, len(x) - window_size + 1, hop_size)]
    # frames = [x]
    S0 = []
    S1 = []
    dic = {0: "eng", 1: "hin"}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for val_lang in tqdm(executor.map(processFrames, frames)):
            S0.append(val_lang[0])
            S1.append(val_lang[1])
    return np.array(S0), np.array(S1)


if __name__ == '__main__':
    S0, S1 = pipeline(audio_path)
    print(S0)
    print(S1)
    print("English/ S0 shape: ", S0.shape)
    print("Hindi/ S1 shape: ", S1.shape)
