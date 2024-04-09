#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## loading important libraries
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
import numpy as np
from tqdm import tqdm
import logging 
import torchaudio
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
import torch
from transformers import AutoConfig, Wav2Vec2Processor
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchaudio
import os
from SilenceRemover import *
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
import IPython.display as ipd
import pandas as pd
from Model import *
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random
import fairseq

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

## This path takes a huggingface datasets library type saved directory of all the languages
saved_dataset_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/combined-saved-dataset.hf"
logging.info(f"Using dataset present at location: {saved_dataset_path}")

class HiddenFeatureExtractor:
    def __init__(self, data: DataLoader, gpu_id: int) -> None:
        ##################################################################################################
        ## Important Intializations
        ##################################################################################################
        global saved_dataset_path
        self.gpu_id = gpu_id
        self.data  = data
        self.max_batch_size = 512
        self.isOneSecond = False
        self.repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        # self.model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displce-2sec-finetunedOndev-300M-saved-model_20240218_143551/pthFiles/model_epoch_9"
        self.model_name_or_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/data2vec-aqc/SPRING_INX_data2vec_aqc_SSL.pt"
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
        self.hiddenFeaturesPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/data2vec-aqc/"
        self.givenName = "data2vec-aqc-springlabs-12lang-hiddenFeatures-fast"
        self.frames = 49 if self.isOneSecond else 99
        self.cols = np.arange(0,1024,1)
        self.chunk_size = 16000 if self.isOneSecond else 32000
        # self.processed_dataset_givenName = "combined-saved-dataset-12lang-new-2sec-processed.hf"
        # self.processed_dataset_path  = os.path.join(self.hiddenFeaturesPath,self.processed_dataset_givenName)
        self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
        # self.label_list = ['eng', 'not-eng']
        self.num_labels = len(self.label_list)
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}
        self.input_column = "path"
        self.output_column = "language"
        self.targetDirectory = os.path.join(self.hiddenFeaturesPath,self.givenName)
        self.processor, self.model_wave2vec2, self.feature_extractor = self.load_model(self.model_name_or_path)

        logging.info(f"All hidden features csv will be saved to directory --> {self.targetDirectory}")
        os.makedirs(self.targetDirectory, exist_ok=True)
        logging.info(f"On GPU {self.gpu_id}")
        
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.processor.feature_extractor.return_attention_mask = True ## to return the attention masks

        self.label_list.sort()  # Let's sort it for determinism
        self.num_labels = len(self.label_list)
    
    def load_model(self, path: str):
        logging.info(f"(GPU {self.gpu_id}) Loading model from path: {path}")
        model_wave2vec2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        model_wave2vec2 = model_wave2vec2[0].to(self.gpu_id)
        model_wave2vec2.eval()
        processor = Wav2Vec2Processor.from_pretrained(self.repo_url)
        model_wave2vec2 =  DDP(model_wave2vec2, device_ids=[self.gpu_id])
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
        logging.info("(GPU {self.gpu_id}) Successfully loaded model.")
        return processor, model_wave2vec2, feature_extractor


    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech
    
    def preprocess_batch(self, batch):
        names = [path.split("/")[-1] for path in batch[self.input_column]]
        speech_list = [self.speech_file_to_array_fn(path) for path in batch[self.input_column]]
        target_list = [self.label2id[label] for label in batch[self.output_column]]

        inputs = self.processor(
            speech_list, 
            sampling_rate=self.processor.feature_extractor.sampling_rate, 
            max_length=self.chunk_size, 
            truncation=True
        )

        inputs["labels"] = target_list
        inputs["name"] = names
        inputs["input_values"] = speech_list  # Adding preprocessed audio to the inputs
        return inputs

    def hiddenFeatures(self, batch):
        silencedAndOneSecondAudio = batch['input_values']
        features = self.processor(silencedAndOneSecondAudio, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.gpu_id)
        attention_mask  = features.attention_mask.to(self.gpu_id)
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            hidden_features = self.model_wave2vec2.module.forward(input_values,mask=None ,features_only=True, padding_mask=attention_mask)['x']
            logging.info(f"Shape of batched hidden features {hidden_features.shape}")
        batch["hidden_features"] = hidden_features
        for lang,name, hFeature in zip(batch['labels'], batch['name'], batch['hidden_features']):
            tPath = os.path.join(self.targetDirectory,self.id2label[lang])
            os.makedirs(tPath, exist_ok=True)
            fileName = self.id2label[lang] + "_" + name.split(".")[0] + ".csv"
            if hFeature.shape[1] == 1024:
                df = pd.DataFrame(hFeature.cpu(),columns=self.cols)
                df.to_csv(os.path.join(tPath,fileName), encoding="utf-8", index=False)
        return batch
    
    def helper(self):
        for idx, batch in enumerate(self.data):
            logging.info(f"Processing batch {idx} on GPU {self.gpu_id}")
            df_len = len(batch['name'])
            # Assuming you have a memory limit and need to batch the data
            num_batches = (df_len + self.max_batch_size - 1) // self.max_batch_size

            for i in tqdm(range(num_batches)):
                start_idx = i * self.max_batch_size
                end_idx = min((i + 1) * self.max_batch_size, df_len)  # Ensure end_idx doesn't exceed batch_size
                print(f"On gpu {self.gpu_id}, processing sliced chunk s:{start_idx}, e:{end_idx}")
                batch_subset = {key: value[start_idx:end_idx] for key, value in batch.items()}

                # Preprocess the batch
                preprocessed_batch = self.preprocess_batch(batch_subset)
                self.hiddenFeatures(preprocessed_batch)
                logging.info(f"Processed batch {idx}, subset {i}/{num_batches} on GPU {self.gpu_id}")
        logging.info(f"All batches processed. on {self.gpu_id}")

    
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
    global saved_dataset_path, num_indices
    ddp_setup(rank, world_size)
     ## loading the dataset from saved dataset
    dataset = load_from_disk(saved_dataset_path)
    logging.info(f"Datasets loaded succesfully from location {saved_dataset_path}")
    dataset = concatenate_datasets([dataset["train"] ,dataset["validation"]])
    print(f"The provided saved dataset is : {dataset}")

    # # Generate random indices
    # all_indices = list(range(len(dataset)))
    # logging.info(f"Length of present dataset is: {len(all_indices)}")
    # selected_indices = random.sample(all_indices, num_indices)
    
    # # Select subset of the dataset based on random indices
    # dataset = dataset.select(selected_indices)
    # print(f"The selected dataset has {len(dataset)} samples.")

    ngpus_per_node = torch.cuda.device_count() 
    batch_size = int(len(dataset) / ngpus_per_node)
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
    logging.info(f"world size detected is {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    logging.info("All hidden features saved!")
