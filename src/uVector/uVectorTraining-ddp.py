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
from dotenv import load_dotenv
import numpy as np
import pandas as pd
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
import numpy as np
from torch import optim
import glob
from torch.autograd import Variable
from gaussianSmooth import *
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score
import random
import wandb

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

##################################################################################################
## Important Intializations
##################################################################################################
hiddenfeaturesPath  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-2lang-2sec-HiddenFeatures-wave2vec2_full_fast"
batch_size = 1

class MyDataset(Dataset):
    def __init__(self):
        global hiddenfeaturesPath
        self.file_paths= []
        self.label_names = ['eng','not-eng']
        self.label2id={label: i for i, label in enumerate(self.label_names)}
        self.id2label={i: label for i, label in enumerate(self.label_names)}
        self.look_back1 = 20
        self.look_back2  = 50

        for lang in self.label2id.keys():
            for f in glob.glob(os.path.join(hiddenfeaturesPath,lang) + '/*.csv'):
                self.file_paths.append(f)

        # self.file_paths = random.sample(self.file_paths, min(num_samples, len(self.file_paths)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return self.read_input(file_path)
    
    ## This funct reads the hidden features as given by HiddenFeatrues csv and 
    ## prepares it for input to the network
    def read_input(self, f):
        df = pd.read_csv(f, usecols=list(range(0,1024)))
        dt = df.astype(np.float32)

        if dt.shape[0] <= self.look_back2:
            mul_len = int((self.look_back2 + 1)/dt.shape[0]) + 1
            # dt = dt.iloc[np.tile(np.arange(dt.shape[0]), mul_len)]
            dt = pd.concat([dt]*mul_len, ignore_index=True)
            
        X = np.array(dt)
        
        Xdata1=[]
        Xdata2=[] 
        Ydata1 =[]
        
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        np.place(std, std == 0, 1)
        X = (X - mu) / std 
        f1 = os.path.splitext(f)[0]     
        
        #### Change when target level (value) changed
        f1 = os.path.split(f)[1]     
        lang = f1.split('_')[0]   

        ### target label (output)
        Y1 = self.label2id[lang]
        Y2 = np.array([Y1])
        
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
        Ydata1 = torch.from_numpy(Y2).long()    
        
        return Xdata1,Xdata2,Ydata1

class uVectorTrain:
    def __init__(self,train_dl:  DataLoader, val_dl: DataLoader) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")

        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"

        self.e_dim = 128*2
        self.nc = 2
        self.look_back1= 20
        self.look_back2  = 50
        self.n_epochs = 100

        ## intializing all the models now
        ## load from chekp path
        self.load_path = None
        self.model_lstm1, self.model_lstm2, self.model = self.load_models(self.load_path)

        self.optimizer = optim.SGD(self.model.module.parameters(),lr = 0.001, momentum= 0.9)
        self.loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"displace_2lang-uVectorTraining_{self.timestamp}"
            self.save_model_path = f"displace_2lang-uVectorTraining_saved-model-{self.timestamp}"
            self.root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/uVector"
            self.save_model_path = os.path.join(self.root,self.save_model_path)
            self.pth_path = f"{self.save_model_path}/pthFiles"
            self.chkpt_path = f"{self.save_model_path}/chkpt"
            self.eval_path = f"{self.save_model_path}/evaluations"
            # Create the folder if it doesn't exist
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
                os.makedirs(self.pth_path)
                os.makedirs(self.chkpt_path)
                os.makedirs(self.eval_path)
                logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{self.save_model_path}'.")
            # ## signing in to wanddb
            # load_dotenv()
            # secret_value_1 = os.getenv("wandb")

            # if secret_value_1 is None:
            #     logging.error(f"Please set Environment Variables properly for wandb login. Exiting.")
            #     sys.exit(1)
            # else:
            #     # Initialize Wandb with your API keywandb
            #     wandb.login(key=secret_value_1)
            #     self.run = wandb.init(name = self.wandb_run_name, project="huggingface")
            #     logging.info("Login to wandb succesfull!")
    
    def save_model(self, epoch:int):
        logging.info("Saving the models snapshot.")
        snapshot = {
            "lstm_model1":self.model_lstm1.module.state_dict(),
            "lstm_model2":self.model_lstm2.module.state_dict(),
            "main_model":self.model.module.state_dict(),
            "epoch":epoch,
        }
        torch.save(snapshot, os.path.join(self.pth_path,f"allModels_epoch_{epoch%6}"))
        logging.info(f"Snapshot checkpointed successfully at location {self.pth_path} with number {epoch%6}")
    
    def load_models(self, path :str):
        # Load the saved models' state dictionaries
        model1 = LSTMNet(self.e_dim).to(self.gpu_id)
        model2 = LSTMNet(self.e_dim).to(self.gpu_id)
        model3 = CCSL_Net(model1, model2, self.nc, self.e_dim).to(self.gpu_id)

        model1 = DDP(model1, device_ids=[self.gpu_id])
        model2 = DDP(model2, device_ids=[self.gpu_id])
        model3 = DDP(model3, device_ids=[self.gpu_id])

        if path is not None:
            snapshot = torch.load(path)
            model1.module.load_state_dict(snapshot["lstm_model1"], strict=False)
            model2.module.load_state_dict(snapshot["lstm_model2"], strict=False)
            model3.module.load_state_dict(snapshot["main_model"], strict=False)
            logging.info("Models loaded successfully from the saved path.")

        return model1, model2, model3


    def run_epoch(self, epoch: int):
        logging.info(f"On gpu: {self.gpu_id}")

        ### Training
        train_cost = 0.0
        train_preds = []
        train_gts = []

        self.model.train()
        self.model_lstm1.train()
        self.model_lstm2.train()

        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm(self.train_dl, desc=f"Processing Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

        for X1, X2, Y1 in batch_iterator:
            torch.cuda.empty_cache()

            X1 = X1[0].to(self.gpu_id)
            X2 = X2[0].to(self.gpu_id)
            Y1 = Y1[0].to(self.gpu_id)

            self.optimizer.zero_grad()  # Zero gradients
            lang_op = self.model.module.forward(X1, X2)
            err_l = self.loss_lang(lang_op, Y1)
            err_l.backward()
            self.optimizer.step()

            train_cost += err_l.item()

            predictions = torch.argmax(lang_op, dim=1).cpu().numpy()
            train_preds.extend(predictions)
            train_gts.extend(Y1.cpu().numpy())

        # Calculate mean accuracy and loss after the epoch
        mean_acc = accuracy_score(train_gts, train_preds)
        mean_loss = train_cost / len(self.train_dl)

        ### Validation
        val_cost = 0.0
        val_preds = []
        val_gts = []

        self.model.eval()
        self.model_lstm1.eval()
        self.model_lstm2.eval()
        logging.info(f"GPU {self.gpu_id}, Evaluating now...")

        with torch.no_grad():
            for i, (X1, X2, Y1) in enumerate(self.val_dl):
                X1 = X1[0].to(self.gpu_id)
                X2 = X2[0].to(self.gpu_id)
                Y1 = Y1[0].to(self.gpu_id)

                lang_op = self.model.module.forward(X1, X2)
                err_l = self.loss_lang(lang_op, Y1)

                val_cost += err_l.item()

                predictions = torch.argmax(lang_op, dim=1).cpu().numpy()
                val_preds.extend(predictions)
                val_gts.extend(Y1.cpu().numpy())

            # Calculate mean accuracy and loss for validation
            val_mean_acc = accuracy_score(val_gts, val_preds)
            val_mean_loss = val_cost / len(self.val_dl)

        # Print combined training and validation stats
        if self.gpu_id == 0:
            logging.info('(GPU {}) Epoch {}: Training Loss {:.5f}, Training Accuracy {:.5f} | Validation Loss {:.5f}, Validation Accuracy {:.5f}'.format(self.gpu_id, epoch, mean_loss, mean_acc, val_mean_loss, val_mean_acc))

    def train(self):
        logging.info("Starting the training!")
        for epoch in range(self.n_epochs):
            self.run_epoch(epoch)
            if self.gpu_id == 0:
                # saving the model
                self.save_model(epoch)

            
    def run(self):
        self.train()


def prepare_dataloader(dataset: Dataset):
    global batch_size
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True)
    )

def main():
    global saved_dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"

     ## loading the dataset from saved dataset
    df = MyDataset()

    # Split the main dataset into training and validation sets
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(df, [train_size, test_size])
    logging.info(f"The train dataset len is {len(train_dataset)}")
    logging.info(f"The validataion dataset len is {len(val_dataset)}")

    train_dataloader = prepare_dataloader(train_dataset)
    val_dataloader = prepare_dataloader(val_dataset)
    uvector = uVectorTrain(train_dataloader, val_dataloader)
    ## train
    uvector.run()
    
    return

if __name__ == '__main__':
    # Setup distributed training
    init_process_group(backend='nccl')

    # Train the model
    main()

    # Clean up distributed training
    destroy_process_group()