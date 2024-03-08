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

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
torch.cuda.empty_cache()


##################################################################################################
## Important Intializations
##################################################################################################
audioPath = "/nlsasfs/home/nltm-st/sujitk/NNDL"
### supervised dev dataset
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale"
wantDER = False
ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm"
saved_dataset_path =  "/nlsasfs/home/nltm-st/sujitk/NNDL"

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
    os.environ["MASTER_PORT"] = "30000"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class LanguageDiarizer:
    def __init__(self, test_data: DataLoader, gpu_id: int) -> None:
        global audioPath, sys_rttmPath
        self.test_data = test_data
        self.gpu_id = gpu_id
        self.nc = 2
        self.look_back1 = 21
        self.IP_dim = 1024 * self.look_back1
        self.window_size = 48000
        self.hop_length_seconds = 0.25
        self.gauss_window_size = 21
        self.max_batch_size = math.ceil(256/(math.ceil(self.window_size/63000)))
        self.sigma = 0.003 * 21
        ## displace 2 lang model trinined on 2sec
        self.xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/xVector-2sec-saved-model-20240218_123206/pthFiles/modelEpoch0_xVector.pth"
        # self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displce-2sec-finetunedOndev-300M-saved-model_20240218_143551/pthFiles/model_epoch_9"

        ### 12 lang finetuned model 2sec
        # self.xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/combined_12lang_2sec-300M_xVector_saved-model-20240226_221808/pthFiles/modelEpoch0_xVector.pth"
        # self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/combined2-300M-saved-model_20240219_133424/pthFiles/model_epoch_2"

        ## olde 11 lang finetuned on 1 sec
        # self.xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/xVectorResults/modelEpoch0_xVector.pth"
        # self.offline_model_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"

        ### for terminator last model
        # self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_20240301_191527/pthFiles/model_epoch_0"
        self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_20240301_191527/pthFiles/model_epoch_4"

        self.audioPath = audioPath
        self.resultDERPath = sys_rttmPath
        self.model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name_or_path)


        logging.info(f"On GPU {self.gpu_id}")
        self.model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(self.offline_model_path).to(gpu_id)
        self.model_wave2vec2 = DDP(self.model_wave2vec2, device_ids=[gpu_id])


        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.processor.feature_extractor.return_attention_mask = True
        self.model_xVector = X_vector(self.IP_dim, self.nc).to(gpu_id)
        self.optimizer =  optim.Adam(self.model_xVector.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
        self.loss_lang = torch.nn.CrossEntropyLoss()
        self.manual_seed = random.randint(1,10000)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        self.label_names = ['eng','not-eng']
        # self.label_names = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi','pun', 'tam', 'tel']
        self.label2id={label: i for i, label in enumerate(self.label_names)}
        self.id2label={i: label for i, label in enumerate(self.label_names)}
        self.indices_to_extract =  [0, 1]
        os.makedirs(self.resultDERPath, exist_ok=True)
        self.targetDirectory = saved_dataset_path
        logging.info(f"All features csv will be saved to directory --> {self.targetDirectory}")
        os.makedirs(self.targetDirectory, exist_ok=True)
        self.count = 0

        try:
            self.model_xVector.load_state_dict(torch.load(self.xVectormodel_path, map_location=torch.device(gpu_id)), strict=False)
            self.model_xVector = DDP(self.model_xVector, device_ids=[gpu_id])
        except Exception as err:
            print("Error is: ",err)
            logging.error("No, valid/corrupted TDNN saved model found, Aborting!")
            sys.exit(0)

    def run_command(self, command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return None

    def preProcessSpeech(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech_array = resampler(speech_array).squeeze().numpy()
        return speech_array

    ## function to store the hidden feature representation from the last layer of wave2vec2
    def getHiddenFeatures(self,frames):
        features = self.processor(frames, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.gpu_id)
        attention_mask  = features.attention_mask.to(self.gpu_id)
        try:
            # Pass attention_mask to the model to prevent attending to padded values
            with torch.no_grad():
                hidden_features = self.model_wave2vec2.module.extract_hidden_states(input_values, attention_mask=attention_mask)
        except Exception as err:
            print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
        return hidden_features

    def inputTDNN(self,hidden_features):
        #### Function to return data (vector) and target label of a csv (MFCC features) file
        X = hidden_features.reshape(-1,1024)
        Xdata1 = []
        for i in range(0,len(X)-self.look_back1,1):    #High resolution low context        
            a = X[i:(i+self.look_back1),:]  
            b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
            Xdata1.append(b)
        Xdata1 = np.array(Xdata1)    
        Xdata1 = torch.from_numpy(Xdata1).float() 
        return Xdata1

    def extractHE(self,input_array):
        print(f"Input array is {input_array}")
        # Select columns eng and hindi
        selected_columns = input_array[:,self.indices_to_extract]
        print(f"Input extracted array is {selected_columns}")
        # Apply softmax along the second axis (axis=1)
        softmax_result = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), axis=1, arr=selected_columns)
        print(f"Input array is {softmax_result}")
        return softmax_result

    # Additional function for model inference
    def modelInference(self, x, path: str):
        file_name = path.split("/")[-1].split(".")[0]
        X_val = torch.from_numpy(np.vectorize(self.inputTDNN, signature='(n,m)->(p,q)')(x)).to(self.gpu_id)  # Returns (#frames, 28, 21504)
        X_val = Variable(X_val, requires_grad=False)
        # Set the model to evaluation mode
        self.model_xVector.eval()
        # Forward pass through the model
        with torch.no_grad():
            x_Vector_Embeddings = self.model_xVector.module.extract_x_vec(X_val)
        for xVector in x_Vector_Embeddings:
            print(f"shape of each xvector: {xVector.shape}")
            tPath = os.path.join(self.targetDirectory,file_name)
            os.makedirs(tPath, exist_ok=True)
            fileName = file_name + "_" + str(self.count) + "_.csv"
            self.counter = self.counter +1 
            # Save the 512-dimensional vector as a CSV without headers
            np.savetxt(os.path.join(tPath, fileName), xVector.cpu().numpy().reshape(1,-1), delimiter=",")
            self.count  = self.count +1
        return None

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
        predictions = []
        for i in tqdm(range(0, len(frames), self.max_batch_size)):
            batch_frames = frames[i:i+self.max_batch_size]
            batch_hidden_features = self.getHiddenFeatures(batch_frames).cpu().numpy()
            print(f"Intermedidate shape {self.gpu_id}: {batch_hidden_features.shape}")
            batch_predictions = self.modelInference(batch_hidden_features, path)
            predictions.append(batch_predictions)
        # print(f"Finall concatenated predictipns: len={len(predictions)}, \n {predictions}")

        print(f"Processed all the frames of given audio {path}!")
        # Concatenate the predictions from all minibatches
        S0 = np.concatenate([p[:, 0] for p in predictions])
        S1 = np.concatenate([p[:, 1] for p in predictions])
        # print(f"Shape of S0: {S0.shape} and S1: {S1.shape}")
        return S0, S1

        
    def predictOne(self,audioPath):
        name = audioPath.split("/")[-1].split(".")[0]
        S0, S1 = self.pipeline(audioPath)

    def helper(self):
        generated_rttms = [self.predictOne(path) for paths in self.test_data for path in paths]

    
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
    
