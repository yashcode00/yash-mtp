#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessarys and libraries
import sys
import math
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from Model import *
from SilenceRemover import *
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import subprocess
import re
from tqdm import tqdm
import torchaudio
import os
import sys
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
import numpy as np
from typing import Any, Dict, Union, Tuple
from torch import optim
import random
from torch.autograd import Variable
# import torch.distributed
from gaussianSmooth import *
import logging
from datetime import datetime
import fairseq


# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
hop_length_seconds = 0.25

def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model_wave2vec2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    model_wave2vec2 = model_wave2vec2[0].to(device)
    model_wave2vec2.eval()
    processor = Wav2Vec2Processor.from_pretrained("yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained("yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor")
    logging.info("Successfully loaded model.")
    return processor, model_wave2vec2, feature_extractor



def load_models(path :str):
    # Load the saved models' state dictionaries
    snapshot = torch.load(path)
    model1 = LSTMNet(e_dim).to(device)
    model2 = LSTMNet(e_dim).to(device)
    model3 = CCSL_Net(model1, model2, nc, e_dim).to(device)

    if path is not None:
        model1.load_state_dict(snapshot["lstm_model1"], strict=False)
        model2.load_state_dict(snapshot["lstm_model2"], strict=False)
        model3.load_state_dict(snapshot["main_model"], strict=False)
        logging.info("Models loaded successfully from the saved path.")
    else:
        logging.error("NO saved model dict found for u-vector!!")

    return model1, model2, model3

e_dim = 128*2
look_back1= 20
look_back2  = 50
window_size = 32000
max_batch_size = 512
model_wave2vec_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/SPRING_INX_wav2vec2_SSL.pt"
model_uvector_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector/wave2vec2_12lang-uVectorTraining_saved-model-20240310_041313/pthFiles/allModels_epoch_5"
label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
num_labels = len(label_list)
nc = num_labels
label2id={label: i for i, label in enumerate(label_list)}
id2label={i: label for i, label in enumerate(label_list)}
input_column = "path"
output_column = "language"
processor ,model_wave2vec2, featrue_extractor = load_model(model_wave2vec_path)
dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/Bhashini_Test_Data"

## the u-vector model
model_lstm1, model_lstm2, model_uVector = load_models(model_uvector_path)
optimizer = optim.SGD(model_uVector.parameters(),lr = 0.001, momentum= 0.9)
loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')

target_sampling_rate = processor.feature_extractor.sampling_rate
processor.feature_extractor.return_attention_mask = True

## function to store the hidden feature representation from the last layer of wave2vec2
def getHiddenFeatures(frames):
    features = processor(frames, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask  = features.attention_mask.to(device)
    try:
        # Pass attention_mask to the model to prevent attending to padded values
        with torch.no_grad():
             hidden_features = model_wave2vec2.forward(input_values,mask=None ,features_only=True, padding_mask=attention_mask)['x']
    except Exception as err:
        print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
    return hidden_features


## This funct reads the hidden features as given by HiddenFeatrues csv and 
## prepares it for input to the network
def inputUvector(hidden_features):
    X = hidden_features.reshape(-1,1024)

    Xdata1=[]
    Xdata2=[] 
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std   
    
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,2):     #Low resolution long context       
        b=X[i+1:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)
    
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()   
    
    return Xdata1,Xdata2

# Additional function for model inference
def modelInference( x):
    # Vectorize the input data
    X1, X2 = np.vectorize(inputUvector, signature='(n,m)->(p,q,m),(a,s,m)')(x)
    batch_size = x.shape[0]
    outputs = []
    for i in range(batch_size):
        X1_i = Variable(torch.from_numpy(X1[i]).to(device), requires_grad=False)
        X2_i = Variable(torch.from_numpy(X2[i]).to(device), requires_grad=False)

        print(f"Shape of processed input for uvector: X1: {X1_i.shape} and X2: {X2_i.shape}")

        # Set the model to evaluation mode
        model_uVector.eval()

        # Forward pass through the model
        with torch.no_grad():
            val_lang_op = model_uVector.forward(X1_i, X2_i)  # Access module attribute for DDP-wrapped model
            val_lang_op = val_lang_op.detach().cpu().numpy()

        print(f"Model Output: shape: {val_lang_op.shape} and \n {val_lang_op}")
        outputs.append(np.argmax(val_lang_op))

    outputs = np.concatenate(outputs, axis=0)  # Combine outputs for all elements in the batch
    # Vectorize the outputs
    print(f"uvector model final output: shape: {outputs.shape} and \n {outputs}")
    return outputs

def preProcessSpeech(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech_array = resampler(speech_array).squeeze().numpy()
    return speech_array[0]

def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]

def pipeline( path):
    # Step 1: Preprocess the audio by removing silence
    x = preProcessSpeech(path)
    print(f"path: {path} len of x {len(x)}")
    # Step 2: Generate overlapping frames
    hop_size = int(hop_length_seconds * target_sampling_rate)
    frames = [x[i:i+window_size] for i in range(0, len(x) - window_size + 1, hop_size)]
    print(f"frame leng: {len(frames)}")
    if len(frames[-1]) < 100:
        print(f"Last element has small length of {len(frames[-1])} while it shall be {len(frames[0])}, Dropping!")
        frames.pop()
    
    # Step 3: Process each batch separately and perform model inference immediately
    predictions = []
    for i in tqdm(range(0, len(frames), max_batch_size)):
        batch_frames = frames[i:i+max_batch_size]
        batch_hidden_features = getHiddenFeatures(batch_frames).cpu().numpy()
        # print(f"Intermedidate shape {device}: {batch_hidden_features.shape}")
        batch_predictions = modelInference(batch_hidden_features)
        predictions.append(batch_predictions)
    print(f"Finall concatenated predictipns: len={len(predictions)}, \n {predictions}")
    print(f"Prediction for {path.split('/')[-1]} is {most_frequent(np.array(predictions).reshape(-1))}")

    print(f"Processed all the frames of given audio {path}!")

for audio in tqdm(os.listdir(dir)):
    pipeline(os.path.join(dir, audio))

logging.info("Done!")
