#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

################## used Library  ############################################################
import torch
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from dotenv import load_dotenv
import os 
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import random
from torch import optim
from tdnn import TDNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb
from Model import *
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import logging
from datetime import datetime
from accelerate import DistributedDataParallelKwargs
from transformers import default_data_collator



# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],mixed_precision= 'fp16')

##################################################################################################
## Important Intializations
##################################################################################################

isOneSecond = False ## chunk size is 16000
frames = 49 if isOneSecond else 99
chunk_size = 16000 if isOneSecond else 32000
nc = 12 # Number of language classes 
n_epoch = 20 # Number of epochs
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
trigger_times = 0
patience = 6
batch_size = 128
label_names = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
label2id={label: i for i, label in enumerate(label_names)}
id2label={i: label for i, label in enumerate(label_names)}
load_model_from_path= None

if accelerator.is_main_process:
    ### making output save folders 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"combined_12lang_2sec-300M_xVector_{timestamp}"
    save_model_path = f"combined_12lang_2sec-300M_xVector_saved-model-{timestamp}"
    root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn"
    save_model_path = os.path.join(root,save_model_path)
    pth_path = f"{save_model_path}/pthFiles"
    chkpt_path = f"{save_model_path}/chkpt"
    eval_path = f"{save_model_path}/evaluations"
    logging.info(f"label2id mapping: {label2id}")
    logging.info(f"id2label mapping: {id2label}")
    # Create the folder if it doesn't exist
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        os.makedirs(pth_path)
        os.makedirs(chkpt_path)
        logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{save_model_path}'.")

train_path  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/combined-new12lang-2sec-HiddenFeatures_full_fast"
result_path = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn'

######################## X_vector ####################
model = X_vector(IP_dim, nc)
optimizer =  optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output


#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)
##################################################################################################
##################################################################################################

class MyDataset(Dataset):
    def __init__(self):
        global train_path
        self.file_paths= []
        for lang in label2id.keys():
            for f in glob.glob(os.path.join(train_path,lang) + '/*.csv'):
                self.file_paths.append(f)

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
        Y1 = np.array([label2id[lang]])

        for i in range(0,len(X)-look_back1,1):    #High resolution low context        
            a = X[i:(i+look_back1),:]  
            b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
            Xdata1.append(b)
        Xdata1 = np.array(Xdata1)    
        Xdata1 = torch.from_numpy(Xdata1).float() 
        Y1 = torch.from_numpy(Y1).long()
        return Xdata1, Y1[0]


df = MyDataset()

# Split the main dataset into training and validation sets
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(df, [train_size, test_size])


train_dataloader = DataLoader(  train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)

val_dataloader = DataLoader(  val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=True)
#### Sample output of the dataloader will have last featrue length constant (21504)
# torch.Size([128, 78, 21504]), torch.Size([128]) for a batch_size of 128 and input frame of 32000 samples at 16kHz

if load_model_from_path is not None:
    model.load_state_dict(torch.load(load_model_from_path), strict=False)
else:
    logging.info("No, saved model found, starting trainig from scratch.")

logging.info("About to start training ddp.")
logging.info('preparing to use accelerate for enhanced distributed training.')

model = model.to(accelerator.device)
device = accelerator.device
accelerator.print("Device of acceleration: ",str(accelerator.device))

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, val_dataloader, model, optimizer
)

## logging into the wandb
## loading env variables
if accelerator.is_main_process:
    load_dotenv()
    secret_value_1 = os.getenv("wandb")

    if secret_value_1 is None:
        logging.error(f"Please set Environment Variables properly for wandb login. Exiting.")
        sys.exit(1)
    else:
        # Initialize Wandb with your API keywandb
        wandb.login(key=secret_value_1)
        run = wandb.init(name = wandb_run_name, project="huggingface")
        logging.info("Login to wandb succesfull!")
    # Save the starting state
    accelerator.save_state(chkpt_path)
    logging.info("Started checkpointing")

for e in range(n_epoch):
    train_loss = []
    validation_loss = []
    x, y = np.array([]), np.array([])
    ## getting model ready for training
    model.train() 
    for X1, Y1 in tqdm(train_dataloader):  
        X1 , Y1 = X1.to(device), Y1.to(device)    
        # print(f"Input shape: {X1.shape} and {Y1.shape}"): Input shape: torch.Size([128, 78, 21504]) and torch.Size([128])
        model.zero_grad()
        preds = model.forward(X1)
        loss = loss_lang(preds, Y1)
        accelerator.backward(loss)
        train_loss.append(torch.mean(accelerator.gather(loss)))
        optimizer.step()
        preds = preds.argmax(dim=-1)
        # Track predictions and ground truth labels
        predictions, references = accelerator.gather_for_metrics((preds, Y1))
        try:
            x = np.concatenate((x,predictions.cpu().numpy()),axis=0)
            y = np.concatenate((y,references.cpu().numpy()),axis=0)
        except Exception as err:
            logging.error("Error Converting to np and processing the x and y: ",err)

    # Calculate train accuracy
    train_accuracy = accuracy_score(x,y)
    train_loss = sum(train_loss)/len(train_loss)

    ## Validation starts from here
    x, y = np.array([]), np.array([])
    # Set model to evaluation mode
    model.eval()
    for X1, Y1 in tqdm(val_dataloader):  
        X1 , Y1 = X1.to(device), Y1.to(device)   
        model.zero_grad()
        preds = model.forward(X1)
        loss = loss_lang(preds, Y1)
        accelerator.backward(loss)
        validation_loss.append(torch.mean(accelerator.gather(loss)))
        optimizer.step()
        preds = preds.argmax(dim=-1)
        # Track predictions and ground truth labels
        predictions, references = accelerator.gather_for_metrics((preds, Y1))
        try:
            x = np.concatenate((x,predictions.cpu().numpy()),axis=0)
            y = np.concatenate((y,references.cpu().numpy()),axis=0)
        except Exception as err:
            logging.error("Error Converting to np and processing the x and y: ",err)

    # Calculate train accuracy
    val_accuracy = accuracy_score(x,y)
    validation_loss = sum(validation_loss)/len(validation_loss)

    if accelerator.is_main_process:
        # Log metrics to WandB for this epoch
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": val_accuracy
        })

        # Print epoch, training loss, validation loss, training accuracy, and validation accuracy
        accelerator.print(f"Epoch {e + 1}/{n_epoch}: Train Loss={train_loss}, Validation Loss={validation_loss}, Train Accuracy={train_accuracy:.4f}, Validation Accuracy={val_accuracy:.4f}")

        #### Saving the results and model of each epoch    
        modelTempName =  f"modelEpoch{e%10}_xVector.pth"
        torch.save(model.state_dict(),os.path.join(pth_path, modelTempName)) # saving the model parameters 
        accelerator.print(f"Checkpointed model at {os.path.join(pth_path,modelTempName)}")