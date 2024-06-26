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
from datasets import Dataset
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
# import torch.distributed
from gaussianSmooth import *
import logging
from datetime import datetime

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

torch.cuda.empty_cache()

##################################################################################################
## Important Intializations
##################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

wantDER = False
batch_size = 128 # Set your desired batch size
max_batch_size = 256
nc = 2 # Number of language classes 
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
window_size = 48000
hop_length_seconds = 0.25  # Desired hop length in seconds
## parameters for gaussian smoothing 
gauss_window_size = 21  # A good starting value for the window size
sigma = 0.003*21  # A reasonable starting value for sigma
xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/xVector-2sec-saved-model-20240218_123206/pthFiles/modelEpoch0_xVector.pth"
resultDERPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults"
resultFolderGivenName = f"displace-test-dev-finetuned-oneval-predicted-rttm-{nc}-lang-{window_size}"
resultDERPath = os.path.join(resultDERPath, resultFolderGivenName) 
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_eval_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"
der_metric_txt_name = "displace-der-metrics-21Feb2024"
derScriptPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/findDER.py"

### Intializing models
## for wave2vec2
model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displce-2sec-finetunedOndev-300M-saved-model_20240218_143551/pthFiles/model_epoch_9"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(offline_model_path).to(device)
target_sampling_rate = processor.feature_extractor.sampling_rate

processor.feature_extractor.return_attention_mask = True
# print("The processor configuration is as follow: ",processor)

### for x-vector
model_xVector = X_vector(IP_dim, nc)
# model.cuda()
optimizer =  optim.Adam(model_xVector.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output
#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

label_names = ['eng','not-eng']
label2id={'eng': 0, 'not-eng': 1}
id2label={0: 'eng', 1: 'not-eng'}
print(f"label2id mapping: {label2id}")
print(f"id2label mapping: {id2label}")

##################################################################################################
##################################################################################################


def create_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
create_output_folder(resultDERPath)

try:
    ## only use map location for non cuda devices
    model_xVector.load_state_dict(torch.load(xVectormodel_path, map_location=torch.device(device)), strict=False)
except Exception as err:
    print("Error is: ",err)
    logging.error("No, valid/corrupted TDNN saved model found, Aborting!")
    sys.exit(0)

def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return None

def diarize(S0, S1):
    # Step 1: Gaussian smooth all the input
    S0_smoothed = gauss_smoothen(S0, sigma, gauss_window_size)
    S1_smoothed = gauss_smoothen(S1, sigma, gauss_window_size)

    # Step 2: Take the sign of the signed difference
    x = np.sign(S0_smoothed - S1_smoothed)

    # Step 3: Take the first-order difference
    x_diff = np.diff(x)

    # Step 4: Find final change points and language labels
    x_abs = np.abs(x_diff)
    change_points = np.where(x_abs == 1)[0]
    lang_labels = np.where(S0 > S1, 0, 1)  # Assuming S0 represents language 0 and S1 represents language 1

    if len(change_points) == 0:
        # Dummy language labels
        lang_labels = np.zeros((len(change_points) + 1,), dtype=int)
        return change_points, lang_labels

    try:
        # Extract language labels based on change points
        lang_labels = [np.argmax(np.bincount(lang_labels[:math.ceil(change_points[0])]))]
    except Exception as e:
        print("Language label extraction error:", e)
        lang_labels = np.zeros((len(change_points) + 1,), dtype=int)
        return change_points, lang_labels

    for i in range(1, len(change_points) + 1):
        lang_labels.append(1 - lang_labels[i - 1])

    return change_points, lang_labels


# def diarize(S0,S1):
#     ## Step 1: gaussian smooth all the input 
#     S0  = np.array(gauss_smoothen(S0, sigma, gauss_window_size))
#     S1 = np.array(gauss_smoothen(S1, sigma, gauss_window_size))

#     ## Step 2: take signum of signed differece 
#     x = np.sign(S0 - S1)
#     ## Step 3: take first order difference
#     x = np.diff(x)
#     ## Step 4: Find final cp's and the language labels
#     x = 0.50*x
#     x = np.abs(x)
#     x = np.where(x == 1)
#     SL = []
#     for i in range(0, len(S0)):
#         if S0[i]>S1[i]:
#             SL.append(0)
#         else:
#             SL.append(1)
#     if len(x[0]) == 0:
#         ## dummy lang labels
#         lang_labels = np.zeros((len(x[0])+1,), dtype=int)
#         return x, lang_labels
#     try: 
#         lang_labels = [np.argmax(np.bincount(SL[:math.ceil(x[0][0])]))]
#     except Exception as e:
#         print("Lang lable extraction error: ",SL,"\n",e)
#         lang_labels = np.zeros((len(x[0])+1,), dtype=int)
#         return x, lang_labels
#     for i in range(1,len(x[0])+1):
#         lang_labels.append(1-lang_labels[i-1])
#     return x, lang_labels

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
        # Pass attention_mask to the model to prevent attending to padded values
        with torch.no_grad():
            hidden_features = model_wave2vec2.extract_hidden_states(input_values, attention_mask=attention_mask)
    except Exception as err:
        print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
    return hidden_features

def inputTDNN(hidden_features):
    #### Function to return data (vector) and target label of a csv (MFCC features) file
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
    selected_columns = input_array[:, [0, 1]]
    # Apply softmax along the second axis (axis=1)
    softmax_result = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), axis=1, arr=selected_columns)
    return softmax_result

# Additional function for model inference
def modelInference(x):
    X_val = torch.from_numpy(np.vectorize(inputTDNN, signature='(n,m)->(p,q)')(x))  # Returns (#frames, 28, 21504)
    X_val = Variable(X_val, requires_grad=False)
    model_xVector.eval()  # Set the model to evaluation mode
    val_lang_op = model_xVector.forward(X_val)
    val_lang_op = val_lang_op.detach().cpu().numpy()
    val_lang_op = np.vectorize(extractHE, signature='(n,m)->(n,p)')(val_lang_op)
    return val_lang_op

def pipeline(path, max_batch_frames=max_batch_size):
    # Step 1: Preprocess the audio by removing silence
    x = preProcessSpeech(path)
    
    # Step 2: Generate overlapping frames
    hop_size = int(hop_length_seconds * target_sampling_rate)
    frames = [x[i:i+window_size] for i in range(0, len(x) - window_size + 1, hop_size)]
    if len(frames[-1]) < 100:
        print(f"Last element has small length of {len(frames[-1])} while it shall be {len(frames[0])}, Dropping!")
        frames.pop()
    
    # Step 3: Batch processing for feature extraction
    results = []
    for i in range(0, len(frames), max_batch_frames):
        batch_frames = frames[i:i+max_batch_frames]
        batch_hidden_features = getHiddenFeatures(batch_frames).cpu().numpy()
        results.append(batch_hidden_features)
    
    # Concatenate results of all minibatches
    x = np.concatenate(results, axis=0)

    # Step 4: Model inference and output processing
    ## Feeds the hidden features to tdnn and gets the output
    val_lang_op = modelInference(x)

    return val_lang_op[:, 0], val_lang_op[:, 1]


def generate_rttm_file(name,cp, predicted_labels, total_time):
    rttm_content = ""
    # Add the start time at 0
    start_time = 0
    tolang = {0:"L1",1:"L2"}
    for i in range(len(cp)):
        end_time = cp[i]
        # Calculate duration for each segment
        duration = end_time - start_time
        # Generate RTTM content
        rttm_content += f"LANGUAGE {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
        # rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> <NA> <NA>\n"

        # Update start time for the next segment
        start_time = end_time
    
    ## add last entry
    duration = total_time - start_time
    i = len(cp)
    # rttm_content += f"Language {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> <NA> <NA>\n"
    rttm_content += f"LANGUAGE {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
    output_rttm_filename = f"{name}_LANGUAGE_sys.rttm"
    targetPath = os.path.join(resultDERPath,output_rttm_filename)

    # Export RTTM file
    with open(targetPath, "w") as rttm_file:
        rttm_file.write(rttm_content)
    return targetPath

# Define a custom sorting key function to extract the numeric part from the filenames
def numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

def findCumulativeDER():
    ##List all DER files in the result directory
    der_files = [os.path.join(resultDERPath, filename) for filename in os.listdir(resultDERPath) if (filename.startswith(der_metric_txt_name) and filename.endswith(".txt"))]
    
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
        logging.error("No DER files found or unable to calculate average DER.")
    
def predictOne(audioPath):
    name = audioPath.split("/")[-1].split(".")[0]
    S0, S1 = pipeline(audioPath)
    x, lang_labels = diarize(S0, S1)
    x = (x[0]*hop_length_seconds)+0.5
    ## now generating rttm for this
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(audioPath)
    # Get the duration in seconds
    duration_in_seconds = waveform.size(1) / sample_rate
    return generate_rttm_file(name, x,lang_labels, duration_in_seconds)

def helper(batch):
    generated_rttms = [predictOne(path) for path in batch["audio_path"]]
    batch["sys_rttm"] = generated_rttms
    return batch

if __name__ == '__main__':
    torch.cuda.empty_cache()
    ## ground truth rttms
    paths = [os.path.join(audioPath,audio) for audio in os.listdir(audioPath)]

    # Create a dataset using Hugging Face datasets library
    dataset = Dataset.from_dict({"audio_path": paths})

    dataset = dataset.map(
        helper,
        batched=True,
        batch_size=1,
        # num_proc=6
        )
    
    sys_rttm = dataset["sys_rttm"]
    sys_rttm = sorted(sys_rttm, key=numeric_part)
    print(f"System generated rttm files: \n{sys_rttm}")
    logging.info(f"Want DER variable set to  {wantDER}")
    if wantDER:
        ref_rttm = [os.path.join(ref_rttmPath,filename) for filename in os.listdir(ref_rttmPath)]
        ref_rttm = sorted(ref_rttm, key=numeric_part)
        print(f"The refernce/ground truth rttm file: \n{ref_rttm}")
        total_batches = math.ceil(float(len(ref_rttm)*1.0)/float(batch_size))
        start_idx = 0
        for batches in tqdm(range(total_batches)):
            end_idx = min(start_idx+batch_size, len(ref_rttm))
            ## finding files
            temp_sys_rttm = sys_rttm[start_idx:end_idx]
            temp_ref_rttm = ref_rttm[start_idx:end_idx]
            temp_sys_rttm = " ".join(temp_sys_rttm)
            temp_ref_rttm = " ".join(temp_ref_rttm)
            # Finding the DER
            command = f'python {derScriptPath} -r {temp_ref_rttm} -s {temp_sys_rttm}'
            # Run the command
            output = run_command(command)
            # print("Command ran: ",command)
            der_filename  = os.path.join(resultDERPath,f"{der_metric_txt_name}-{batches}.txt")
            with open(der_filename, 'w') as f_out:
                f_out.write(output)
            if end_idx == len(ref_rttm):
                break
            start_idx = end_idx +1
        
        der  = findCumulativeDER()
        logging.info(f"The average DER is: {der}")
    logging.info("Evaluation Completed Succesfully!")
    logging.info(f"Please check {resultDERPath} for system generated rttm.")


