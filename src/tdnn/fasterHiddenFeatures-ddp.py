#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## loading important libraries
import sys
import math
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging 
import shutil
import torchaudio
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import AutoConfig, Wav2Vec2Processor
import numpy as np
from typing import Any, Dict, Union
import torch
from packaging import version
import pandas as pd
from tqdm import tqdm
import torchaudio
import os
import IPython.display as ipd
from SilenceRemover import *
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
import IPython.display as ipd
import numpy as np
import pandas as pd
from Model import *
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

saved_dataset_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/combined-saved-dataset.hf"

class HiddenFeatureExtractor:
    def __init__(self, data: DataLoader, gpu_id: int) -> None:
        ##################################################################################################
        ## Important Intializations
        ##################################################################################################
        global saved_dataset_path
        self.gpu_id = gpu_id
        self.data  = data
        self.isOneSecond = False
        self.repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/combined2-300M-saved-model_20240219_133424/pthFiles/model_epoch_0"
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
        self.hiddenFeaturesPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/"
        self.hiddenFeatures_givenName = "combined-saved-dataset-12lang-new.hf"
        self.givenName = "combined-saved-dataset-12lang-new-2sec-HiddenFeatures_full_fast"
        self.frames = 49 if self.isOneSecond else 99
        self.cols = np.arange(0,1024,1)
        self.chunk_size = 16000 if self.isOneSecond else 32000
        self.processed_dataset_givenName = "combined-saved-dataset-12lang-new-2sec-processed.hf"
        self.processed_dataset_path  = os.path.join(self.hiddenFeaturesPath,self.processed_dataset_givenName)
        self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}
        self.input_column = "path"
        self.output_column = "language"
        self.targetDirectory = os.path.join(self.hiddenFeaturesPath,self.givenName)
        os.makedirs(self.targetDirectory, exist_ok=True)


        self.config = AutoConfig.from_pretrained(self.repo_url)
        self.processor = Wav2Vec2Processor.from_pretrained(self.repo_url)

        logging.info(f"On GPU {self.gpu_id}")
        self.model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(self.model_name_or_path).to(gpu_id)
        self.model_wave2vec2 = DDP(self.model_wave2vec2, device_ids=[gpu_id])

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.processor.feature_extractor.return_attention_mask = True ## to return the attention masks

        self.label_list.sort()  # Let's sort it for determinism
        self.num_labels = len(self.label_list)

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def hiddenFeatures(self, batch):
        silencedAndOneSecondAudio = batch['input_values']
        features = self.processor(silencedAndOneSecondAudio, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.device)
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            hidden_features = self.model.extract_hidden_states(input_values)
        batch["hidden_features"] = hidden_features
        for lang,name, hFeature in zip(batch['language'], batch['name'], batch['hidden_features']):
            tPath = os.path.join(self.targetDirectory,lang)
            fileName = lang + "_" + name.split(".")[0] + ".csv"
            if hFeature.shape[1] == 1024:
                df = pd.DataFrame(hFeature.cpu(),columns=self.cols)
                df.to_csv(os.path.join(tPath,fileName), encoding="utf-8", index=False)
        return batch

    def helper(self):
        for idx, batch in enumerate(self.data):
            print(f"ID: {idx} and batch : {batch}")
            break
        sys.exit(0)
    
    def run(self):
        logging.info(f"Finding hidden features of the dataset on gpu {self.gpu_id}")
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
    global saved_dataset_path
    ddp_setup(rank, world_size)
     ## loading the dataset from saved dataset
    dataset = load_from_disk(saved_dataset_path)
    print(f"Datasets loaded succesfully from location {saved_dataset_path}")
    dataset = concatenate_datasets([dataset["train"] ,dataset["validation"]])
    print(f"The dataset is : {dataset}")
    ngpus_per_node = torch.cuda.device_count() 
    batch_size = int(len(dataset) / ngpus_per_node)
    batch_size = 3
    dataset.set_format("torch")
    ## Loading the paths of the audios into a torch dataset
    logging.info(f"The batch size per gpu will be {batch_size}")
    test_data = prepare_dataloader(dataset, batch_size=batch_size)
    evaluater = HiddenFeatureExtractor(test_data, rank)
    res = evaluater.run()
    print(f"Result from {rank} is \n{res}")
    destroy_process_group()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(f"world size detected is {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    logging.info("All hidden featrues saved!")


# # print(train_dataset)
# if __name__ == "__main__": 
#     targetDirectory = os.path.join(hiddenFeaturesPath,givenName)
#     if os.path.exists(targetDirectory) and os.path.isdir(targetDirectory):
#         shutil.rmtree(targetDirectory)

#      ### Creating folder for saving hidden featrues as csv files
#     if givenName not in os.listdir(hiddenFeaturesPath):
#         os.makedirs(targetDirectory)
#         ##  first lets crewat all the subdirectories
#         for lang in label_list:
#             new_path = CreateIfNot(targetDirectory, lang)
#     else:
#         print(f"{os.path.join(hiddenFeaturesPath,givenName)}: Folder already exists!")
        
#     print(f"A classification problem with {num_labels} classes: {label_list}")
#     if os.path.exists(processed_dataset_path):
#         logging.info("Filtered and processed dataset detected, skipping preprocessing.")
#         logging.info(f"Loading from {processed_dataset_path}")
#         train_dataset = load_from_disk(processed_dataset_path)
#         print(f"Dataset: {train_dataset}")
#         print(f"Length of each input_values is : {len(train_dataset[0]['input_values'])}")
#     else:
#         ## loading the dataset from saved dataset
#         dataset = load_from_disk(final_path)
#         print("Datasets loaded succesfully!!")
#         train_dataset = dataset["train"]
#         eval_dataset = dataset["validation"]
#         print("Final/Train: ",train_dataset)
#         print("Validation: ",eval_dataset)

#         train_dataset = concatenate_datasets([train_dataset ,eval_dataset])
#         train_dataset = train_dataset.map(
#                             preprocess_function,
#                             batch_size=256,
#                             batched=True,
#                             num_proc=400,
#                             # keep_in_memory=True
#                             load_from_cache_file=True
#         )

#         # ## filtering the dataset
#         # new_dataset  = []
#         # print("Filtering Dataset....")
#         # for row in tqdm(train_dataset):
#         #     l = len(row['input_values'])
#         #     if l < 900:
#         #         print("skipping length too short: ",l)
#         #         continue 
#         #     new_dataset.append(row)

#         # train_dataset =  Dataset.from_list(new_dataset)
#         ## filtering the dataset
#         print("Filtering Dataset....")
#         train_dataset = train_dataset.filter(lambda example: len(example['input_values']) >= 900)


#         print("Saving Processed dataset... ")
#         try:
#             train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,processed_dataset_givenName))
#             logging.info(f"Saved proceesed dataset to {os.path.join(hiddenFeaturesPath,processed_dataset_givenName)}")
#         except:
#             logging.error("Unable to save to the disk: ", Exception)

#     train_dataset = train_dataset.map(
#         hiddenFeatures,
#         batch_size=512,
#         batched=True,
#         # num_proc=400,
#         # keep_in_memory=True
#         # load_from_cache_file=True
#     )

#     print("Saving hiddenFeatures dataset... ")
#     try:
#         train_dataset.save_to_disk(os.path.join(hiddenFeaturesPath,hiddenFeatures_givenName))
#         print("Saved hiddenFeatures to the directory: ",os.path.join(hiddenFeaturesPath,hiddenFeatures_givenName))
#     except:
#         logging.error("Unable to save to the disk: ", Exception)

#     logging.info("Hidden Features extracted succesfully.")

    



