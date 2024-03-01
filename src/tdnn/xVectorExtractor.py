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
from torch import optim
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
import torch
from transformers import AutoConfig, Wav2Vec2Processor
import numpy as np
import pandas as pd
import glob
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

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
num_indices = 50000

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
saved_dataset_path =  "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-2lang-2sec-HiddenFeatures-wave2vec2_full_fast"
logging.info(f"Using dataset present at location: {saved_dataset_path}")

class x_Vector_extractor:
    def __init__(self, data: DataLoader, gpu_id: int) -> None:
        ##################################################################################################
        ## Important Intializations
        ##################################################################################################
        global saved_dataset_path
        self.gpu_id = gpu_id
        self.data  = data
        self.counter = 0
        self.nc = 2
        self.look_back1 = 21 # range
        self.IP_dim = 1024*self.look_back1 # number of input dimension
        self.trigger_times = 0
        self.patience = 6
        self.max_batch_size = 512
        self.xVectormodel_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn/xVector-2sec-saved-model-20240218_123206/pthFiles/modelEpoch0_xVector.pth"
        self.cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
        self.hiddenFeaturesPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/x-vector-embeddings"
        # self.hiddenFeatures_givenName = "combined-saved-dataset-12lang-new.hf"
        self.givenName = "displace-2lang-2sec-xVector-embeddings_full_fast"
        self.cols = np.arange(0,512,1)
        # self.processed_dataset_givenName = "combined-saved-dataset-12lang-new-2sec-processed.hf"
        # self.processed_dataset_path  = os.path.join(self.hiddenFeaturesPath,self.processed_dataset_givenName)
        # self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
        self.label_list = ['eng', 'not-eng']
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}
        self.input_column = "path"
        self.output_column = "language"
        self.targetDirectory = os.path.join(self.hiddenFeaturesPath,self.givenName)
        logging.info(f"All hidden features csv will be saved to directory --> {self.targetDirectory}")
        os.makedirs(self.targetDirectory, exist_ok=True)

        logging.info(f"On GPU {self.gpu_id}")
        self.model_xVector = X_vector(self.IP_dim, self.nc).to(gpu_id)
        self.optimizer =  optim.Adam(self.model_xVector.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
        self.loss_lang = torch.nn.CrossEntropyLoss()
        self.manual_seed = random.randint(1,10000)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        self.label_list.sort()  # Let's sort it for determinism
        self.num_labels = len(self.label_list)

        try:
            self.model_xVector.load_state_dict(torch.load(self.xVectormodel_path, map_location=torch.device(gpu_id)), strict=False)
            self.model_xVector = DDP(self.model_xVector, device_ids=[gpu_id])
        except Exception as err:
            print("Error is: ",err)
            logging.error("No, valid/corrupted TDNN saved model found, Aborting!")
            sys.exit(0)

    def hiddenFeatures(self, batch):
        input_values = batch["input_values"].to(self.gpu_id)
        # print(f"The shape of the input values is {input_values.shape}") ###torch.Size([512, 78, 21504])
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            x_Vector_Embeddings = self.model_xVector.module.extract_x_vec(input_values)
        batch["xVector"] = x_Vector_Embeddings
        for lang, xVector in zip(batch['labels'], batch['xVector']):
            lang = lang.item()
            tPath = os.path.join(self.targetDirectory,self.id2label[lang])
            os.makedirs(tPath, exist_ok=True)
            fileName = self.id2label[lang] + "_" + str(self.counter) + "_.csv"
            self.counter = self.counter +1 
            # Save the 512-dimensional vector as a CSV without headers
            np.savetxt(os.path.join(tPath, fileName), xVector.cpu().numpy().reshape(1,-1), delimiter=",")
        return batch
    
    def helper(self):
        for idx, batch in enumerate(self.data):
            logging.info(f"Processing batch {idx} on GPU {self.gpu_id}")
            batch , targets = batch
            df_len = len(batch)
            # print(f"batch: {batch} \n target: {targets}")
            # Assuming you have a memory limit and need to batch the data
            num_batches = (df_len + self.max_batch_size - 1) // self.max_batch_size

            for i in tqdm(range(num_batches)):
                start_idx = i * self.max_batch_size
                end_idx = min((i + 1) * self.max_batch_size, df_len)  # Ensure end_idx doesn't exceed batch_size
                print(f"On gpu {self.gpu_id}, processing sliced chunk s:{start_idx}, e:{end_idx}")
                batch_subset = {"input_values":batch[start_idx:end_idx] ,"labels": targets[start_idx:end_idx]}
                # print(f"batch subset: {batch_subset}")
                # Preprocess the batch
                self.hiddenFeatures(batch_subset)
                logging.info(f"Processed batch {idx}, subset {i}/{num_batches} on GPU {self.gpu_id}")
        logging.info(f"All batches processed. on {self.gpu_id}")

    
    def run(self):
        logging.info(f"Finding x-vector embeddings of the dataset on gpu {self.gpu_id}")
        res = self.helper()
        logging.info(f"Task completed on gpu {self.gpu_id} with result as follow\n {res}")
        return res

class MyDataset(Dataset):
    def __init__(self):
        global saved_dataset_path, num_indices
        self.file_paths= []
        self.look_back1 = 21 # range
        self.label_list = ['eng', 'not-eng']
        self.label2id={label: i for i, label in enumerate(self.label_list)}
        self.id2label={i: label for i, label in enumerate(self.label_list)}
        for lang in self.label2id.keys():
            for f in glob.glob(os.path.join(saved_dataset_path,lang) + '/*.csv'):
                self.file_paths.append(f)
        # Select a random shuffled subset of tuples
        self.file_paths = random.sample(self.file_paths, min(len(self.file_paths), num_indices))
        logging.info(f"Datasets loaded succesfully from location {saved_dataset_path}")


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data, label = self.read_input(file_path)
        return data, label
    
    ## This funct reads the hidden features as given by HiddenFeatrues csv and 
    ## prepares it for input to the TDNN
    def read_input(self, path: str):
        df = pd.read_csv(path)
        dt = df.astype(np.float32)
        X = np.array(dt).reshape(-1,1024)
        if X.shape[1] != 1024:
            logging.error("Invalid shape of hidden features (need to be 1024) csv, skipping")
            return None, None
        Xdata1 = []
        f1 = os.path.split(path)[1]     
        lang = f1.split('_')[0] 
        Y1 = np.array([self.label2id[lang]])

        for i in range(0,len(X)- self.look_back1,1):    #High resolution low context        
            a = X[i:(i+self.look_back1),:]  
            b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
            Xdata1.append(b)
        Xdata1 = np.array(Xdata1)    
        Xdata1 = torch.from_numpy(Xdata1).float() 
        Y1 = torch.from_numpy(Y1).long()
        return Xdata1, Y1[0]
    

#### Sample output of the dataloader will have last featrue length constant (21504)

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
    df = MyDataset()
    ngpus_per_node = torch.cuda.device_count() 
    batch_size = int(len(df) / ngpus_per_node)
    ## Loading the paths of the audios into a torch dataset
    logging.info(f"The batch size per gpu will be {batch_size}")
    test_data = prepare_dataloader(df, batch_size=batch_size)
    evaluater = x_Vector_extractor(test_data, rank)
    res = evaluater.run()
    print(f"Result from {rank} is \n{res}")
    destroy_process_group()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    logging.info(f"world size detected is {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    logging.info("All hidden features saved!")
