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
# import fairseq
from datetime import datetime
from pyannote.core import Segment


# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
torch.cuda.empty_cache()


##################################################################################################
## Important Intializations
##################################################################################################
audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_eval_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
### supervised dev dataset
# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/HE_codemixed_audio_SingleSpeakerFemale"
wantDER = False

# audioPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/Audio"
# ref_rttmPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/testDiralisationOutput/rttm"

root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults/u-Vector/spring-labs"
root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults/phase2/u-vector"
resultFolderGivenName = f"wave2vec2-withvad-pretrained-2lang-eval-48000-0.25-predicted-rttm-lang-20-50-displace"
sys_rttmPath = os.path.join(root,resultFolderGivenName)
PYANNOT_SEG_PATH =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/vad_audio_segments"
PYANNOT_SEG_PATH = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_eval_audio_supervised/AUDIO_supervised/vad_audio_segments"

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
        global audioPath, sys_rttmPath, PYANNOT_SEG_PATH
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
        self.pyannot_seg_path = PYANNOT_SEG_PATH


        print(f"the batch size for evaluation (max) is {self.max_batch_size}")
        self.sigma = 0.003 * 21

        ## 12 lang wave2vec2
        # self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/SPRING_INX_wav2vec2_SSL.pt"
        # self.uvector_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector/springlabs-uVectorTraining_saved-model-20240316_180010/pthFiles/allModels_epoch_0"

        ### 2 lang best
        self.offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_20240301_191527/pthFiles/model_epoch_0"
        self.uvector_model_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector/displace_2lang-uVectorTraining_saved-model-20240305_182126/pthFiles/allModels_epoch_3"

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
        #                                                                 config=config , 
        #                                                                     cache_dir=self.cache_dir
        #                                                                 ).to(self.gpu_id)
        model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(path,
                                                                # config=config , 
                                                                    cache_dir=self.cache_dir
                                                                ).to(self.gpu_id)
        # model_wave2vec2.load_state_dict(snapshot, strict=False)
        model_wave2vec2 =  DDP(model_wave2vec2, device_ids=[self.gpu_id])
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
        logging.info(f"(GPU {self.gpu_id}) Successfully loaded wave2vec2 model.")
        return config, processor, model_wave2vec2, feature_extractor

    # def load_model_wave2vec2(self, path: str):
    #     logging.info(f"(GPU {self.gpu_id}) Loading model from path: {path}")
    #     model_wave2vec2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    #     model_wave2vec2 = model_wave2vec2[0].to(self.gpu_id)
    #     model_wave2vec2.eval()
    #     processor = Wav2Vec2Processor.from_pretrained(self.repo_url)
    #     model_wave2vec2 =  DDP(model_wave2vec2, device_ids=[self.gpu_id])
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
    #     logging.info(f"(GPU {self.gpu_id}) Successfully loaded model.")
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
            # logging.info(f"{x} is empty bro, labels generated are: {lang_labels}")
            return x, lang_labels
        try: 
            lang_labels = [np.argmax(np.bincount(SL[:math.ceil(x[0][0])]))]
        except Exception as e:
            # print("Lang lable extraction error: ",SL,"\n",e)
            lang_labels = np.zeros((len(x[0])+1,), dtype=int)
            return x, lang_labels
        for i in range(1,len(x[0])+1):
            lang_labels.append(1-lang_labels[i-1])
        # print("Segment Labels are: ", SL)
        # lang_labels = np.zeros(len(x)+1)
        return x, lang_labels

    ## function to store the hidden feature representation from the last layer of wave2vec2
    def getHiddenFeatures(self,frames):
        features = self.processor(frames, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.gpu_id)
        attention_mask  = features.attention_mask.to(self.gpu_id)
        try:
            # Pass attention_mask to the model to prevent attending to padded values
            with torch.no_grad():
                hidden_features = self.model_wave2vec2.module.extract_hidden_states(input_values, attention_mask=attention_mask)
                # hidden_features =  self.model_wave2vec2.module.forward(input_values,mask=None ,features_only=True, padding_mask=attention_mask)['x']
        except Exception as err:
            print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
        return hidden_features
    
    @staticmethod
    def parseSegFile(segFilepath: str):
        logging.info(f"Reading segments at {segFilepath}")
        """
        Parse Pyannote segments from a file into a list of Segment objects.

        Args:
            segFilepath (str): Path to the file containing Pyannote segment string.

        Returns:
            List of Segment: List of Segment objects parsed from the file.
        """
        segments = []
        # Open the file and read each line
        with open(segFilepath, 'r') as f:
            for line in f:
                # Split each line into start and end times
                start, end = map(float, line.strip().split())
                # Create Segment object and append to the list
                segments.append(Segment(start, end))
        return segments
    
    def audioVad(self, path, segments) :
        ## loading the audio file
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech_array = resampler(speech_array).squeeze().numpy()

        audio_segments = []
        # Iterate over segments
        for start, end in segments:
            # Convert start and end times to sample indices
            start_sample = int(start * self.target_sampling_rate)
            end_sample = int(end * self.target_sampling_rate)
            # Extract segment
            segment_waveform = speech_array[start_sample:end_sample]
            # Create Segment object
            segment = Segment(start, end)
            # Append audio segment and segment object to the list
            audio_segments.append((segment_waveform, segment))
        return audio_segments

    ## This funct reads the hidden features as given by HiddenFeatrues csv and 
    ## prepares it for input to the network
    def inputUvector(self,hidden_features):
        seq , _ = hidden_features.shape
        if seq <= max(self.look_back1, self.look_back2):
            look_back2 = int(seq/2) if int(seq/2) != 0 else 2
            look_back1 = int(seq/4) if int(seq/4) != 0 else 4
        else:
            look_back1= self.look_back1
            look_back2 = self.look_back2

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

    def extractHE(self,input_array):
        # Select columns eng and hindi
        selected_columns = input_array[:,self.indices_to_extract]
        # Apply softmax along the second axis (axis=1)
        softmax_result = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), axis=1, arr=selected_columns)
        return softmax_result

    # Additional function for model inference
    def modelInference(self, x):
        # Vectorize the input data
        logging.info(f"shape of input to {x.shape}")
        X1, X2 = np.vectorize(self.inputUvector, signature='(n,m)->(p,q,m),(a,s,m)')(x)
        # logging.info(f"shape of input to x-vector/u-vector: {X1.shape} and {X2.shape}")
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            X1_i = Variable(torch.from_numpy(X1[i]).to(self.gpu_id), requires_grad=False)
            X2_i = Variable(torch.from_numpy(X2[i]).to(self.gpu_id), requires_grad=False)

            # print(f"Shape of processed input for uvector: X1: {X1_i.shape} and X2: {X2_i.shape}")

            # Set the model to evaluation mode
            self.model_uVector.eval()

            # Forward pass through the model
            with torch.no_grad():
                val_lang_op = self.model_uVector.module.forward(X1_i, X2_i)  # Access module attribute for DDP-wrapped model
                val_lang_op = val_lang_op.detach().cpu().numpy()

            # print(f"Model Output: shape: {val_lang_op.shape} and \n {val_lang_op}")
            outputs.append(val_lang_op)

        outputs = np.concatenate(outputs, axis=0)  # Combine outputs for all elements in the batch
        # Vectorize the outputs
        outputs = np.vectorize(self.extractHE, signature='(n,m)->(n,p)')(outputs)
        # print(f"uvector model final output: shape: {outputs.shape} and \n {outputs}")
        return outputs


    def pipeline(self, x):
        # Step 1: Generate overlapping frames
        hop_size = int(self.hop_length_seconds * self.target_sampling_rate)

        frames = []
        for i in range(0, len(x), hop_size):
            e = min(i+self.window_size, len(x))
            frames.append(x[i:e])

        if len(frames) == 0:
            logging.error(f"This arr cause the issue: {len(x)}")
            sys.exit(0)
        if len(frames[-1]) < 600:
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

        # print(f"Processed all the frames of given audio {path}!")
        # Concatenate the predictions from all minibatches
        S0 = np.concatenate([p[:, 0] for p in predictions])
        S1 = np.concatenate([p[:, 1] for p in predictions])
        # print(f"Shape of S0: {S0.shape} and S1: {S1.shape}")
        return S0, S1

    def generate_rttm_file(self, cp,name, predicted_labels, total_time, start_time):
        rttm_content = ""
        tolang = {0:"L1",1:"L2"}
        for i in range(len(cp)):
            end_time = cp[i]
            # Calculate duration for each segment
            duration = end_time - start_time
            # Generate RTTM content
            rttm_content += f"LANGUAGE {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
            # Update start time for the next segment
            start_time = end_time
        
        ## add last entry
        duration = total_time - start_time
        if duration > 0:
            i = len(cp)
            rttm_content += f"LANGUAGE {name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {tolang[predicted_labels[i]]} <NA> <NA>\n"
        return rttm_content
        
    def predictOne(self,audio_path: str, segFile_path: str):
        # step 1: get all chunks usning vad
        segments_loaded = LanguageDiarizer.parseSegFile(segFile_path)
        x = self.audioVad(audio_path, segments_loaded)
        name = audio_path.split("/")[-1].split(".")[0]

        rttm_sys = ""
    
        ## step 2: loop through all segments and make prediction
        for audio_array, interval in tqdm(x):
            start, end = interval
            if abs(round(end - start,2)) <= 1:
                continue
            # if len(audio_array) < 1000:
            #     continue
            logging.info(f"GPU: {self.gpu_id}, ON start: {start} - end: {end} , and {len(audio_array)}")
            S0, S1 = self.pipeline(audio_array)
            x, lang_labels = self.diarize(S0, S1)
            x = (x[0]*self.hop_length_seconds)+((self.window_size/16000) - self.hop_length_seconds)*0.50
            ## adding offsest of strt time to all the elments
            x = [float(val + start) for val in x]
            ## now generating rttm for this
            # Get the duration in seconds
            duration_in_seconds = len(audio_array) / self.target_sampling_rate
            if len(x) == 0:
                # Determine the final label using majority votingm,
                final_labels = np.where(S0 > S1, 0, 1)
                arr = np.argmax(np.bincount(final_labels))
                rttm_sys += self.generate_rttm_file(x,name, [arr], duration_in_seconds, start)
            else:
                rttm_sys += self.generate_rttm_file(x,name, lang_labels, duration_in_seconds, start)
        
        print("*"*100)
        print(f"Processed all chunks of the audio at path {audio_path}")
        
        ## Step3: compoiling and generating final rttm
        output_rttm_filename = f"{name}_LANGUAGE_sys.rttm"
        targetPath = os.path.join(self.resultDERPath,output_rttm_filename)

        # Export RTTM file
        with open(targetPath, "w") as rttm_file:
            rttm_file.write(rttm_sys)
        return targetPath


    def helper(self):
        generated_rttms = []
        for paths in self.test_data:
            for path in paths:
                audio_name = path.split(".")[0].split("/")[-1]
                seg_file_path = os.path.join(self.pyannot_seg_path, f"{audio_name}.pyannote.segment")
                generated_rttms.append(self.predictOne(path, seg_file_path)) 
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
    