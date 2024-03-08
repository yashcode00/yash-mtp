#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

## importing all the neccesary modules
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from tqdm import tqdm
import torchaudio
import os
from datasets import  load_from_disk, concatenate_datasets
from transformers.file_utils import ModelOutput
import torch
from datetime import datetime 
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
# is_apex_available
import torch
from huggingface_hub import login
import numpy as np
import wandb
from Model import *
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
import logging 
from dotenv import load_dotenv

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# disable_caching()
dataset_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/combined-saved-dataset.hf"
per_device_train_batch_size = 128
per_device_eval_batch_size = 256

##############################################################################################################################
### Main Distributed Training Code for finetuning wave2vec2
##############################################################################################################################
class MyDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.repo_url = "facebook/wav2vec2-xls-r-300m"
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
        self.processor_tokenizer_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.processor = Wav2Vec2Processor.from_pretrained(self.processor_tokenizer_url, cache_dir=self.cache_dir)
        self.processor.feature_extractor.return_attention_mask = True

        self.input_column = "path"
        self.output_column = "language"
        self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi','pun', 'tam', 'tel']
        self.label_list.sort()
        self.num_labels = len(self.label_list)
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}
        self.target_sampling_rate = 16000
        self.chunk_size = 32000
        self.dataset = self.loadDataset()

        
    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def preprocess_function(self, example):
        speech = self.speech_file_to_array_fn(example[self.input_column])
        label = self.label2id[example[self.output_column]]
        inputs = self.processor(
            speech, sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt", 
            max_length = self.chunk_size,
            truncation = True,
            padding='max_length',
        )  
        inputs['input_values'] = torch.flatten(inputs['input_values'])
        inputs['attention_mask'] = torch.flatten(inputs['attention_mask'])
        inputs["labels"] = label
        return inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.preprocess_function(self.dataset[idx])

    def loadDataset(self):
        # loading from saved dataset
        dataset = load_from_disk(self.path)
        dataset = concatenate_datasets([dataset["train"],dataset["validation"]])
        logging.info("Datasets loaded succesfully.")
        return dataset

class wave2vec2Finetune():
    def __init__(self,train_dl:  DataLoader, val_dl: DataLoader) -> None:
        global dataset_path, per_device_eval_batch_size,per_device_train_batch_size
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")
        if self.gpu_id == 0:
            logging.info(f"The len of the train and val dataloader is: \n Train :{len(self.train_dl)} \n Val: {len(self.val_dl)}")
        self.wandb_run_id = None

        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK e   nvironment variable not set"
        self.per_device_train_batch_size=per_device_train_batch_size,
        self.per_device_eval_batch_size=per_device_eval_batch_size,

        ##################################################################################################
        ## Important Intializations
        ##################################################################################################
        self.base_directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/"
        # self.repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.repo_url = "facebook/wav2vec2-xls-r-300m"
        self.processor_tokenizer_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
        self.model_offline_path = self.repo_url

        # We need to specify the input and output column
        self.input_column = "path"
        self.output_column = "language"
        self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi','pun', 'tam', 'tel']
        self.label_list.sort()
        self.num_labels = len(self.label_list)
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}

        self.n_epochs = 500
        self.chunk_size = 32000 ## the audio chunk size that is used for finetuinng like in this case a  :i.e. 32000 -> 2 sec chunks samplec at 16kHz
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
        self.dataset_path= dataset_path

        ## perform this operation only once on master gpu with global rank 0
        if self.gpu_id == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"wave2vec2-12lang-300M-saved-model_{timestamp}"
            self.save_model_path = f"wave2vec2-12lang-300M-saved-model_{timestamp}"
            self.save_model_path = os.path.join("/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2",save_model_path)
            self.chkpt_path = f"{save_model_path}/chkpt"
            self.pth_path = f"{save_model_path}/pthFiles"
            self.eval_path = f"{save_model_path}/evaluations"
            # Create the folder if it doesn't exist
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
                os.makedirs(chkpt_path)
                os.makedirs(pth_path)
                os.makedirs(eval_path)
                logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{save_model_path}'.")
            
            ## for huggingface and wandb sync  
            load_dotenv()
            secret_value_0 = os.getenv("hugging_face")
            secret_value_1 = os.getenv("wandb")

            if secret_value_0 is None or secret_value_1 is None:
                logging.error(f"Please set Environment Variables properly. Exiting.")
                sys.exit(1)
            else:
                login(secret_value_0)
                logging.info("Logged into hugging face successfully!")
                # Initialize Wandb with your API keywandb
                wandb.login(key=secret_value_1)
                if self.wandb_run_id is not None: 
                    self.run = wandb.init(name = wandb_run_name,id = self.wandb_run_id, project="lid-1")
                else:
                    self.run = wandb.init(name = wandb_run_name, project="lid-1")

        ## loading the model
        self.feature_extractor, self.processor, self.model, self.optimizer = self.load_model()
        self.target_sampling_rate = self.feature_extractor.sampling_rate

    def load_model(self):
        ###############################################################################################################
        ### Now fetching all data and componenets of the specified model
        ###############################################################################################################

        ## loading the wave2vec2 feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.repo_url , cache_dir=self.cache_dir)
        ### loading the processor and tokenizer contained inside it
        pooling_mode = "mean"
        # config
        config = AutoConfig.from_pretrained(
            self.repo_url,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            finetuning_task="wav2vec2_clf",
            cache_dir=self.cache_dir,
        )
        setattr(config, 'pooling_mode', pooling_mode)
        ## Loading the processor for wav2vec2
        processor = Wav2Vec2Processor.from_pretrained(self.processor_tokenizer_url, cache_dir=self.cache_dir)
        ## loading the main model
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            self.model_offline_path,
            config=config , 
            cache_dir=self.cache_dir
        ).to(self.gpu_id)
        ## for transfer learning
        model.freeze_feature_extractor()
        # Instantiate optimizer
        optimizer = AdamW(params=model.parameters(), lr=3e-5)

        model =  DDP(model, device_ids=[self.gpu_id], find_unused_parameters=True)
        
        return feature_extractor, processor, model, optimizer

    def run_epoch(self, epoch):
        logging.info(f"On gpu: {self.gpu_id}")
        train_loss = []
        val_loss = []
        final_train_loss = -1
        final_val_loss = -1
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm(self.train_dl, desc=f"Processing Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
        self.model.train()

        x = np.array([])
        y = np.array([])
        for batch in batch_iterator:
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(self.gpu_id)
            outputs = self.model(**batch)
            loss = outputs.loss
            # loss = loss / gradient_accumulation_steps
            train_loss.append(loss)
            loss.backward()
            self.optimizer.step()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            try:
                x = np.concatenate((x,predictions.cpu().numpy()),axis=0)
                y = np.concatenate((y,references.cpu().numpy()),axis=0)
            except Exception as err:
                logging.error("Error Converting to np and processing the x and y: ",err)
            self.optimizer.zero_grad()

        final_train_loss = sum(train_loss)/len(train_loss)
        train_accuracy = accuracy_score(x,y)

        # saving the model
        self.save_model(epoch)
        
        print(f" (GPU {self.gpu_id}) Evaluating Wait...")
        x = np.array([])
        y = np.array([])
        batch_iterator = tqdm(self.val_dl, desc=f"Processing Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
        self.model.eval()
        for batch in batch_iterator:
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(self.gpu_id)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            loss = outputs.loss
            val_loss.append(self.loss)
            predictions, references = predictions, batch["labels"]
            try:
                x = np.concatenate((x,predictions.cpu().numpy()),axis=0)
                y = np.concatenate((y,references.cpu().numpy()),axis=0)
            except Exception as err:
                logging.error("Error Converting to np and processing the x and y: ",err)
        final_val_loss = sum(val_loss)/len(val_loss)

        if self.gpu_id == 0:
            try:
                result = classification_report(x, y, target_names=self.label_list)
                # logging.info(result)
                # Additional information to include with the report
                additional_info = os.path.join(self.eval_path,f"eval_epoch{epoch}.txt")
                # Save the report with additional information to a text file
                with open(additional_info, 'w') as f:
                    f.write(result)
            except:
                logging.error("Error in evaluate metric compute: ",str(Exception))
            # Use accelerator.logging.info to logging.info only on the main process.
            val_accuracy = accuracy_score(x,y)
             ## send the metrics to wancdb
            try:
                # Log metrics to WandB for this epoch
                wandb.log({
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "train_loss": final_train_loss,
                    "val_loss": final_val_loss,
                })

            except Exception as err:
                logging.error("Not able to log to wandb, ", err)
            logging.info(f"(GPU [{self.gpu_id}])Epoch {epoch+1}/{self.n_epochs}: train_loss: {final_train_loss} val_loss: {final_val_loss} Val_Accuracy:{val_accuracy}")

    
    def save_model(self, epoch: int):
        logging.info(f"(GPU {self.gpu_id}) <- Saving Model ->")
        snapshot = {
            "wave2vec2": self.model.module.state_dict(),
            "wand_run_id": self.run.id
        }

        torch.save(snapshot, os.path.join(self.pth_path,f"modelinfo_epoch_{epoch%15}"))
        logging.info(f"Snapshot checkpointed successfully at location {self.pth_path} with number {epoch%15}")     

    def train(self):
        logging.info("Starting the training!")
        for epoch in range(self.n_epochs):
            self.run_epoch(epoch)

def prepareDataloader(dataset_path: str):
    global per_device_eval_batch_size,per_device_train_batch_size
    processor_tokenizer_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
    processor = Wav2Vec2Processor.from_pretrained(processor_tokenizer_url)
    # data_collator = DataCollatorCTCWithPadding(processor=processor)

    ## loading the dataset from saved dataset
    df = MyDataset(dataset_path)
    # Split the main dataset into training and validation sets
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(df, [train_size, test_size])
    logging.info(f"Original Dataset: \n Train: {len(train_dataset)} \n Val: {len(val_dataset)}")
    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        # collate_fn=data_collator, 
        batch_size=per_device_train_batch_size,
        drop_last=True,
        
        sampler=DistributedSampler(train_dataset, shuffle=True)
    )
    val_dataloader = DataLoader(
        val_dataset, 
        shuffle=False, 
        # collate_fn=data_collator, 
        batch_size=per_device_train_batch_size,
        drop_last=True,
        sampler=DistributedSampler(val_dataset, shuffle=True)
    )

    return train_dataloader, val_dataloader



def main():
    global dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    train_dl, val_dl = prepareDataloader(dataset_path)
    wave2vec2 = wave2vec2Finetune(train_dl, val_dl)
    ## train
    wave2vec2.train()

if __name__ == '__main__':
    # Setup distributed training
    init_process_group(backend='nccl')

    # Train the model
    main()

    # Clean up distributed training
    destroy_process_group()
