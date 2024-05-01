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
audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_eval_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
# ### supervised dev dataset
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale"
wantDER = False

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/Audio"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm"

root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults/phase2/u-vector"
resultFolderGivenName = f"test-wave2vec2-displace-2lang-dev-40000-0.25-predicted-rttm-lang-20-50"
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
        self.window_size = 48000
        self.hop_length_seconds = 0.25
        self.gauss_window_size = 21
        self.max_batch_size = math.ceil(256/(math.ceil(self.window_size/63000)))
        self.repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"


        print(f"the batch size for evaluation (max) is {self.max_batch_size}")
        self.sigma = 0.003 * 21

        self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_20240301_191527/pthFiles/model_epoch_0"
        self.uvector_model_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector/displace_2lang-uVectorTraining_saved-model-20240305_182126/pthFiles/allModels_epoch_3"

        # ## 12 lang wave2vec2
        # self.offline_model_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/wave2vec2-12lang-300M-saved-model_20240308_181251/pthFiles/modelinfo_epoch_14"
        # self.uvector_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector/wave2vec2_12lang-uVectorTraining_saved-model-20240310_041313/pthFiles/allModels_epoch_2"


        # ## 2 lang displace
        # self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/Phase2/wave2vec2-2lang-finetunedOnrttm_300M-saved-model_20240416_010420/pthFiles/modelinfo_epoch_6"
        # self.uvector_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/Phase2/uVector/displace-2lang-finetunedOnrttm-uVectorTraining-20240418_083125/pthFiles/allModels_epoch_3"

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
        ## loading all the models into memory
        ### wave2vec2
        _, self.processor, self.model_wave2vec2, self.feature_extractor = self.load_model_wave2vec2(self.offline_model_path)


        ## the u-vector model
        self.model_lstm1, self.model_lstm2, self.model_uVector = self.load_models(self.uvector_model_path)
        self.optimizer = optim.SGD(self.model_uVector.module.parameters(),lr = 0.001, momentum= 0.9)
        self.loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')


        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.processor.feature_extractor.return_attention_mask = True
        os.makedirs(self.resultDERPath, exist_ok=True)

    def load_models(self, path :str):
        # Load the saved models' state dictionaries
        snapshot = torch.load(path)
        model1 = LSTMNet(self.e_dim).to(self.gpu_id)
        model2 = LSTMNet(self.e_dim).to(self.gpu_id)
        model3 = CCSL_Net(model1, model2, self.nc, self.e_dim).to(self.gpu_id)

        model1 = DDP(model1, device_ids=[self.gpu_id])
        model2 = DDP(model2, device_ids=[self.gpu_id])
        model3 = DDP(model3, device_ids=[self.gpu_id])

        if path is not None:
            model1.module.load_state_dict(snapshot["lstm_model1"], strict=False)
            model2.module.load_state_dict(snapshot["lstm_model2"], strict=False)
            model3.module.load_state_dict(snapshot["main_model"], strict=False)
            logging.info("Models loaded successfully from the saved path.")
        else:
            logging.error("NO saved model dict found for u-vector!!")

        return model1, model2, model3
    
    def load_model_wave2vec2(self, path: str):
        # snapshot = torch.load(path, map_location=torch.device('cpu'))["wave2vec2"]
        logging.info(f"(GPU {self.gpu_id}) Loading wave2vec2 model from path: {path}")
        # config
        config = AutoConfig.from_pretrained(
            self.repo_url,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            finetuning_task="wav2vec2_clf",
            cache_dir=self.cache_dir,
        )
        processor = Wav2Vec2Processor.from_pretrained(self.repo_url)
        # model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained("facebook/wav2vec2-xls-r-300m",
                                                                        # config=config , 
                                                                        #     cache_dir=self.cache_dir
                                                                        # ).to(self.gpu_id)
        model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(path,
                                                                # config=config , 
                                                                    cache_dir=self.cache_dir
                                                                ).to(self.gpu_id)
        # model_wave2vec2.load_state_dict(snapshot, strict=False)
        model_wave2vec2 =  DDP(model_wave2vec2, device_ids=[self.gpu_id])
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
        logging.info("(GPU {self.gpu_id}) Successfully loaded wave2vec2 model.")
        return config, processor, model_wave2vec2, feature_extractor

    # def load_model_wave2vec2(self, path: str):
    #     logging.info(f"(GPU {self.gpu_id}) Loading model from path: {path}")
    #     model_wave2vec2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    #     model_wave2vec2 = model_wave2vec2[0].to(self.gpu_id)
    #     model_wave2vec2.eval()
    #     processor = Wav2Vec2Processor.from_pretrained(self.repo_url)
    #     model_wave2vec2 =  DDP(model_wave2vec2, device_ids=[self.gpu_id])
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
    #     logging.info("(GPU {self.gpu_id}) Successfully loaded model.")
    #     return processor, model_wave2vec2, feature_extractor

    def run_command(self, command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return None
        
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

    ## This funct reads the hidden features as given by HiddenFeatrues csv and 
    ## prepares it for input to the network
    def inputUvector(self,hidden_features):
        X = hidden_features.reshape(-1,1024)

        Xdata1=[]
        Xdata2=[] 
        
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        np.place(std, std == 0, 1)
        X = (X - mu) / std   
        
        for i in range(0,len(X)-self.look_back1,1):    #High resolution low context        
            a=X[i:(i+self.look_back1),:]        
            Xdata1.append(a)
        Xdata1=np.array(Xdata1)

        for i in range(0,len(X)-self.look_back2,2):     #Low resolution long context       
            b=X[i+1:(i+self.look_back2):3,:]        
            Xdata2.append(b)
        Xdata2=np.array(Xdata2)
        
        Xdata1 = torch.from_numpy(Xdata1).float()
        Xdata2 = torch.from_numpy(Xdata2).float()   
        
        return Xdata1,Xdata2

    def extractHE(self,input_array):
        # Select columns eng and hindi
        selected_columns = input_array[:,self.indices_to_extract]
        # Apply softmax along the second axis (axis=1)
        softmax_result = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), axis=1, arr=selected_columns)
        return softmax_result

    # Additional function for model inference
    def modelInference(self, x):
        # Vectorize the input data
        X1, X2 = np.vectorize(self.inputUvector, signature='(n,m)->(p,q,m),(a,s,m)')(x)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            X1_i = Variable(torch.from_numpy(X1[i]).to(self.gpu_id), requires_grad=False)
            X2_i = Variable(torch.from_numpy(X2[i]).to(self.gpu_id), requires_grad=False)

            print(f"Shape of processed input for uvector: X1: {X1_i.shape} and X2: {X2_i.shape}")

            # Set the model to evaluation mode
            self.model_uVector.eval()

            # Forward pass through the model
            with torch.no_grad():
                val_lang_op = self.model_uVector.module.forward(X1_i, X2_i)  # Access module attribute for DDP-wrapped model
                val_lang_op = val_lang_op.detach().cpu().numpy()

            print(f"Model Output: shape: {val_lang_op.shape} and \n {val_lang_op}")
            outputs.append(val_lang_op)

        outputs = np.concatenate(outputs, axis=0)  # Combine outputs for all elements in the batch
        # Vectorize the outputs
        outputs = np.vectorize(self.extractHE, signature='(n,m)->(n,p)')(outputs)
        print(f"uvector model final output: shape: {outputs.shape} and \n {outputs}")
        return outputs


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
            # print(f"Intermedidate shape {self.gpu_id}: {batch_hidden_features.shape}")
            batch_predictions = self.modelInference(batch_hidden_features)
            predictions.append(batch_predictions)
        # print(f"Finall concatenated predictipns: len={len(predictions)}, \n {predictions}")

        print(f"Processed all the frames of given audio {path}!")
        # Concatenate the predictions from all minibatches
        S0 = np.concatenate([p[:, 0] for p in predictions])
        S1 = np.concatenate([p[:, 1] for p in predictions])
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