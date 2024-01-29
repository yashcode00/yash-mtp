#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessary modules and libraries
import sys
import math
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from Model import *
from SilenceRemover import *
import numpy as np
import pandas as pd
import subprocess
import re
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
import torch
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
import numpy as np
from typing import Any, Dict, Union, Tuple
from torch import optim
import random
from torch.autograd import Variable
import concurrent.futures
from gaussianSmooth import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
batch_size = 100  # Set your desired batch size
nc = 11 # Number of language classes 
n_epoch = 200 # Number of epochs
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
path = "/Users/yash/Desktop/MTP-2k23-24"
xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/xVectorResults/modelEpoch0_xVector.pth"
outputFolderRTTM = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/predicted-rttm"
resultDERPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults"
audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale"
ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm"
silencedAndOneSecondAudio_size = 16000
window_size = 16000
hop_length_seconds = 0.25  # Desired hop length in seconds
## parameters for gaussian smoothing 
gauss_window_size = 21  # A good starting value for the window size
sigma = 0.003*21  # A reasonable starting value for sigma
audio_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale/HECodemixedFemale1.wav"
ref_rttm = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm/rttm_HECodemixedFemale1.wav"
derScriptPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/findDER.py"

### Intializing models
## for wave2vec2
model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
target_sampling_rate = processor.feature_extractor.sampling_rate

processor.feature_extractor.return_attention_mask = True
print("The processor configuration is as follow: ",processor)

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
# print("Mask for binary (Hin/Eng) classification: ",mask)

def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return None

def diarize(S0,S1):
    ## Step 1: gaussian smooth all the input 
    S0  = np.array(gauss_smoothen(S0, sigma, gauss_window_size))
    S1 = np.array(gauss_smoothen(S1, sigma, gauss_window_size))

    ## Step 2: take signum of signed differece 
    # print(f"S0-S1: {S0-S1}")
    x = np.sign(S0 - S1)
    # print(f"2. After signum: {x}")
    x1 = []
    ## Step 3: take first order difference
    for i in range(1,len(x)-1):
        x1.append(x[i+1] - x[i])
    x1.append(x[len(x)-1])
    x1 = np.array(x1)
    # print(f"2. After firstorder difference:{x1}")
    x = x1
    ## Step 4: Find final cp's and the language labels
    x = 0.50*x
    # print(f"3. After 0.5*x: {x}")
    x = np.abs(x)
    # print(f"4. After abs: {x}")
    x = np.where(x == 1)
    # print("Change points are: ", x)
    SL = []
    for i in range(0, len(S0)):
        if S0[i]>S1[i]:
            SL.append(0)
        else:
            SL.append(1)

# Traceback (most recent call last):
#   File "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/languageDiarizer.py", line 256, in <module>
#     x, lang_labels = diarize(S0, S1)
#   File "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/languageDiarizer.py", line 134, in diarize
#     lang_labels = [np.argmax(np.bincount(SL[:math.floor(x[0][0])]))]
#   File "/nlsasfs/home/nltm-st/sujitk/miniconda3/envs/wave2vec/lib/python3.9/site-packages/numpy/core/fromnumeric.py", line 1229, in argmax
#     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
#   File "/nlsasfs/home/nltm-st/sujitk/miniconda3/envs/wave2vec/lib/python3.9/site-packages/numpy/core/fromnumeric.py", line 59, in _wrapfunc
#     return bound(*args, **kwds)
# ValueError: attempt to get argmax of an empty sequence
    # try: 
    #     lang_labels = [np.argmax(np.bincount(SL[:math.floor(x[0][0])]))]
    # except Exception as e:
    #     print("Lang lable extraction error: ",SL,"\n",e)
    #     lang_labels = np.zeros()
    # for i in range(len(x[0])-1):
    #     lang_labels.append(np.argmax(np.bincount(SL[math.floor(x[0][i]):math.floor(x[0][i+1])])))

    # if len(x[0])!=1:
    #     lang_labels.append(np.argmax(np.bincount(SL[math.floor(x[0][len(x[0])-1]):])))
    # print("Segment Labels are: ", SL)
    lang_labels = np.zeros(len(x)+1)
    return x, lang_labels

def preProcessSpeech(path):
    speech_array, sampling_rate = torchaudio.load(path)
    # speech_array = torch.frombuffer(RemoveSilence(path),dtype=torch.float32)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech_array = resampler(speech_array).squeeze().numpy()
    return speech_array

## function to store the hidden feature representation from the last layer of wave2vec2
def getHiddenFeatures(frames):
    features = processor(frames, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask  = features.attention_mask.to(device)
    try:
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            hidden_features = model_wave2vec2.extract_hidden_states(input_values, attention_mask=attention_mask)
            # logits = model_wave2vec2(input_values, attention_mask=attention_mask).logits 
            # print("Predictions from wave2vec2: ",id2lang[torch.argmax(logits, dim=-1).detach().cpu().numpy()[0]], end=",")
    except Exception as err:
        print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
    # Return the overall majority prediction
    # print("Hiden features size: ",hidden_features.shape)
    return hidden_features

def inputTDNN(hidden_features):
    #### Function to return data (vector) and target label of a csv (MFCC features) file
    # print("Shape of individual hidden features: ",hidden_features.shape)
    X = hidden_features.reshape(-1,1024)
    Xdata1 = []
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a = X[i:(i+look_back1),:]  
        b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
        Xdata1.append(b)
    Xdata1 = np.array(Xdata1)    
    Xdata1 = torch.from_numpy(Xdata1).float() 
    return Xdata1

def extractHE(input_array):
    # Select columns eng and hindi
    selected_columns = input_array[:, [2, 4]]
    # Apply softmax along the second axis (axis=1)
    softmax_result = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), axis=1, arr=selected_columns)
    return softmax_result

def pipeline(path):
    ## Step 1: preprocees the audio by removing silence
    x = preProcessSpeech(path)
    ## Now, we will just bbreak this audio into multiple overlapping windows 
    # Calculate the hop size in samples
    hop_size = int(hop_length_seconds * target_sampling_rate)  # Adjust 'sample_rate' as needed

    # Generate overlapping frames
    frames = [x[i:i+window_size] for i in range(0, len(x) - window_size + 1, hop_size)]
    if len(frames[-1])<100:
        print(f"Last element has small length of {len(frames[-1])} while it shall be {len(frames[0])}, Dropping!")
        frames.pop()
    ## Step 2: get the hidden featrue for the processed output / here #frames acts as batch size
    x = getHiddenFeatures(frames).cpu().numpy() ### returns something like of shape (#frames, 49,1024)
    # print(f"Shape of hidden features of all the frames: {x.shape}")
    ## Step 3: get the output of TDNN for both eng and hindi
    X_val = torch.from_numpy(np.vectorize(inputTDNN, signature='(n,m)->(p,q)')(x))  ## returns (#frames, 28, 21504)
    # print(f"shape of XX_val: {X_val.shape}")
    X_val = Variable(X_val, requires_grad=False)
    model_xVector.eval()  # Set the model to evaluation mode
    val_lang_op =model_xVector.forward(X_val)
    val_lang_op = val_lang_op.detach().cpu().numpy()
    # print(f"shape of val_lang_op: {val_lang_op.shape}")
    val_lang_op = np.vectorize(extractHE,signature='(n,m)->(n,p)')(val_lang_op)
    # print(f"Predicted language for this window is: ",y)
    ## Step 4: mask the output for all languauge except eng and hindi
    # print(val_lang_op)
    return val_lang_op[:,0], val_lang_op[:,1]

def generate_rttm_file(name,cp, predicted_labels, total_time):
    rttm_content = ""
    # Add the start time at 0
    start_time = 0
    tolang = {0:"English",1:"Hindi"}
    for i in range(len(cp)):
        end_time = cp[i]
        # Calculate duration for each segment
        duration = end_time - start_time
        # Generate RTTM content
        # rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
        rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> <NA> <NA>\n"

        # Update start time for the next segment
        start_time = end_time
    
    ## add last entry
    duration = total_time - start_time
    rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> <NA> <NA>\n"
    # rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
    output_rttm_filename = f"Predicted_{name}.rttm"
    targetPath = os.path.join(outputFolderRTTM,output_rttm_filename)

    # Export RTTM file
    with open(targetPath, "w") as rttm_file:
        rttm_file.write(rttm_content)
    return targetPath

# Define a custom sorting key function to extract the numeric part from the filenames
def numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

def findCumulativeDER():
    ##List all DER files in the result directory
    der_files = [os.path.join(resultDERPath, filename) for filename in os.listdir(resultDERPath) if filename.startswith("eval-der-metrics")]
    
    total_der = 0.0
    total_files = 0

    # Process each DER file and calculate average DER
    for der_file in der_files:
        with open(der_file, 'r') as f_in:
            der_content = f_in.read()
            # Extract the DER value from the content
            floats = re.findall(r'-?\d+\.\d+', der_content)
            if len(floats) >= 1:
                der = float(floats[0])
                total_der += der
                total_files += 1

    if total_files > 0:
        avg_der = total_der / total_files
        return avg_der
    else:
        print("No DER files found or unable to calculate average DER.")

if __name__ == '__main__':
    ## ground truth rttms
    sys_rttm = [os.path.join(ref_rttmPath,filename) for filename in os.listdir(ref_rttmPath)]
    sys_rttm = sorted(sys_rttm, key=numeric_part)
    # ref_rttm = []
    # for audio in tqdm(os.listdir(audioPath)):
    #     actualPath = os.path.join(audioPath,audio)
    #     name = audio.split(".")[0]
    #     S0, S1 = pipeline(actualPath)
    #     # print(S0)
    #     # print(S1)
    #     # print("English/ S0 shape: ", S0.shape)
    #     # print("Hindi/ S1 shape: ", S1.shape)
    #     x, lang_labels = diarize(S0, S1)
    #     # print(x)
    #     # print(lang_labels)
    #     x = (x[0]*hop_length_seconds)+0.5
    #     ## now generating rttm for this
    #     # Load the audio file using torchaudio
    #     waveform, sample_rate = torchaudio.load(actualPath)
    #     # Get the duration in seconds
    #     duration_in_seconds = waveform.size(1) / sample_rate
    #     ref_rttm.append(generate_rttm_file(name, x,lang_labels, duration_in_seconds))
    ref_rttm = [os.path.join(outputFolderRTTM,filename) for filename in os.listdir(outputFolderRTTM)]
    ref_rttm = sorted(ref_rttm, key=numeric_part)
    # print("The true and predicted rttm files are: \n")

    # print(sys_rttm)
    # print(ref_rttm)
    ## joining
    # sys_rttm = " ".join(sys_rttm)
    # ref_rttm = " ".join(ref_rttm)

    # print(sys_rttm)
    # print(ref_rttm)
    total_batches = math.ceil(float(len(sys_rttm)*1.0)/float(batch_size))
    start_idx = 0
    for batches in tqdm(range(total_batches)):
        end_idx = min(start_idx+batch_size, len(sys_rttm))
        ## finding files
        temp_ref_rttm = ref_rttm[start_idx:end_idx]
        temp_sys_rttm = sys_rttm[start_idx:end_idx]
        temp_ref_rttm = " ".join(temp_ref_rttm)
        temp_sys_rttm = " ".join(temp_sys_rttm)
        # Finding the DER
        command = f'python {derScriptPath} -r {temp_ref_rttm} -s {temp_sys_rttm}'
        # Run the command
        output = run_command(command)
        # print("Command ran: ",command)
        der_filename  = os.path.join(resultDERPath,f"eval-der-metrics-old-model-29Jan2024-{batches}.txt")
        with open(der_filename, 'w') as f_out:
            f_out.write(output)
        if end_idx == len(sys_rttm):
            break
        start_idx = end_idx +1
    
    der  = findCumulativeDER()
    print(f"The average DER is: {der}")
    print("Evaluation Completed Succesfully!")



