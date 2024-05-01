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
import whisper

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
torch.cuda.empty_cache()


##################################################################################################
## Important Intializations
##################################################################################################
audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_eval_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
# ### supervised dev dataset
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale"
wantDER = True

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/Audio"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm"

root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults/phase2/u-vector"
resultFolderGivenName = f"openai-whisper-displace-2lang-eval-32000-predicted-rttm-lang"
sys_rttmPath = os.path.join(root,resultFolderGivenName)

class AudioPathDataset(Dataset):
    def __init__(self, file_paths): 
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        return audio_path

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "30100"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class LanguageDiarizer:
    def __init__(self, test_data: DataLoader, gpu_id: int) -> None:
        global audioPath, sys_rttmPath
        self.test_data = test_data
        self.gpu_id = gpu_id
        self.e_dim = 128*2
        self.look_back1= 20
        self.look_back2  = 50
        self.window_size = 32000
        self.hop_length_seconds = 0.25
        self.gauss_window_size = 21
        self.max_batch_size = math.ceil(256/(math.ceil(self.window_size/63000)))
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"


        print(f"the batch size for evaluation (max) is {self.max_batch_size}")
        self.sigma = 0.003 * 21

        ## load the openai whisper model
        self.model = whisper.load_model("base").to(self.gpu_id)

        self.audioPath = audioPath
        self.resultDERPath = sys_rttmPath

        self.label_names = ['eng','not-eng']
        # self.label_names = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi','pun', 'tam', 'tel']
        self.label2id={label: i for i, label in enumerate(self.label_names)}
        self.id2label={i: label for i, label in enumerate(self.label_names)}
        self.num_labels = len(self.label_names)
        self.nc = self.num_labels
        self.indices_to_extract =  [0, 1]
        # self.indices_to_extract =  [2, 4]

        logging.info(f"On GPU {self.gpu_id}")
        self.target_sampling_rate = 16000
        os.makedirs(self.resultDERPath, exist_ok=True)

    def run_command(self, command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return None

    def whisper_call(self, audio_chunk):
        audio_seg = whisper.pad_or_trim(audio_chunk)
        mel_seg = whisper.log_mel_spectrogram(audio_seg).to(self.model.device)
        _, prob_seg = self.model.detect_language(mel_seg)
        #print(f"Speaker {speaker_id}: Detected language: {max(prob_seg, key=prob_seg.get)}")
        return prob_seg["en"], prob_seg["hi"]

        
    def diarize(self,S0,S1):
        ## Step 1: gaussian smooth all the input 
        S0  = np.array(gauss_smoothen(S0, self.sigma, self.gauss_window_size))
        S1 = np.array(gauss_smoothen(S1, self.sigma, self.gauss_window_size))

        ## Step 2: take signum of signed differece 
        # print(f"S0-S1: {S0-S1}")
        x = np.sign(S0 - S1)
        # print(f"2. After signum: {x}")
        x1 = []
        ## Step 3: take first order difference
        for i in range(0,len(x)-1):
            x1.append(x[i+1] - x[i])
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
        if len(x[0]) == 0:
            ## dummy lang labels
            lang_labels = np.zeros((len(x[0])+1,), dtype=int)
            return x, lang_labels
        try: 
            lang_labels = [np.argmax(np.bincount(SL[:math.ceil(x[0][0])]))]
        except Exception as e:
            print("Lang lable extraction error: ",SL,"\n",e)
            lang_labels = np.zeros((len(x[0])+1,), dtype=int)
            return x, lang_labels
        for i in range(1,len(x[0])+1):
            lang_labels.append(1-lang_labels[i-1])
        # print("Segment Labels are: ", SL)
        # lang_labels = np.zeros(len(x)+1)
        return x, lang_labels

    def preProcessSpeech(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech_array = resampler(speech_array).squeeze().numpy()
        return speech_array


    def pipeline(self, path):
        # Step 1: Preprocess the audio by removing silence
        x = self.preProcessSpeech(path)
        # Step 2: Generate overlapping frames
        hop_size = int(self.hop_length_seconds * self.target_sampling_rate)
        frames = [x[i:i+self.window_size] for i in range(0, len(x) - self.window_size + 1, hop_size)]
        if len(frames[-1]) < 100:
            print(f"Last element has small length of {len(frames[-1])} while it shall be {len(frames[0])}, Dropping!")
            frames.pop()
        
        # Step 3: Process each batch separately and perform model inference immediately
        S0 = []
        S1 = []
        for i in tqdm(range(0, len(frames)), desc = f"One gpu: {self.gpu_id} processing: {path}"):
            arr = frames[i]
            s0, s1 = self.whisper_call(arr)
            S0.append(s0)
            S1.append(s1)

        # print(f"Finall concatenated predictipns: len={len(predictions)}, \n {predictions}")

        print(f"Processed all the frames of given audio {path}!")
        S0 = np.array(S0)
        S1 = np.array(S1)
        # print(f"Shape of S0: {S0.shape} and S1: {S1.shape}")
        return S0, S1


    def generate_rttm_file(self,name,cp, predicted_labels, total_time):
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
        targetPath = os.path.join(self.resultDERPath,output_rttm_filename)

        # Export RTTM file
        with open(targetPath, "w") as rttm_file:
            rttm_file.write(rttm_content)
        return targetPath
        
    def predictOne(self,audioPath):
        name = audioPath.split("/")[-1].split(".")[0]
        S0, S1 = self.pipeline(audioPath)
        x, lang_labels = self.diarize(S0, S1)
        x = (x[0]*self.hop_length_seconds)+((self.window_size/16000) - self.hop_length_seconds)*0.50
        ## now generating rttm for this
        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(audioPath)
        # Get the duration in seconds
        duration_in_seconds = waveform.size(1) / sample_rate
        return self.generate_rttm_file(name, x,lang_labels, duration_in_seconds)

    def helper(self):
        generated_rttms = [self.predictOne(path) for paths in self.test_data for path in paths]
        return generated_rttms

    
    def run(self):
        logging.info(f"Evaluating the dataset on gpu {self.gpu_id}")
        res = self.helper()
        logging.info(f"Task completed on gpu {self.gpu_id} with result as follow\n {res}")
        return res

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    ## Loading the paths of the audios into a torch dataset
    paths = [os.path.join(audioPath,audio) for audio in os.listdir(audioPath)]
    ngpus_per_node = torch.cuda.device_count() 
    batch_size = int(len(paths) / ngpus_per_node)
    logging.info(f"The batch size per gpu will be {batch_size}")
    dataset = AudioPathDataset(paths)
    test_data = prepare_dataloader(dataset, batch_size=batch_size)
    evaluater = LanguageDiarizer(test_data, rank)
    res = evaluater.run()
    print(f"Result from {rank} is \n{res}")
    destroy_process_group()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(f"world size detected is {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    logging.info("All test audio successfully evaluated!")
    if wantDER:
        logging.info(f"finding DER as flag wantDER is {wantDER}")
        subprocess.run(["python3", "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/findCumulativeDERfromFiles.py", 
                "--ref_rttm_folder_path", ref_rttmPath,
                "--sys_rttm_folder_path", sys_rttmPath ,
                "--out", sys_rttmPath])
        logging.info(f"DER files are saved at {sys_rttmPath}")