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
from torch.autograd import Variable
from torch import optim
from tdnn import TDNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb
from Model import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import logging
from datetime import datetime

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device is: ",device)

def print_gpu_info():
    print("-"*20)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_capability = torch.cuda.get_device_capability(current_device)
        gpu_info = f"Number of GPUs: {device_count}\nCurrent GPU: {current_device}\nGPU Name: {device_name}\nGPU Compute Capability: {device_capability}"
        print(gpu_info)
        for i in range(device_count):
            print(f"GPU {i} Memory Usage:")
            print(torch.cuda.memory_summary(i))
    else:
        print("No GPU available.")
    print("-"*20)

print_gpu_info()

##################################################################################################
## Important Intializations
##################################################################################################

isOneSecond = False ## chunk size is 16000
frames = 49 if isOneSecond else 99
chunk_size = 16000 if isOneSecond else 32000
nc = 2 # Number of language classes 
n_epoch = 10 # Number of epochs
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
trigger_times = 0
patience = 6
batch_size = 128
label_names = ['eng','not-eng']
label2id={'eng': 0, 'not-eng': 1}
id2label={0: 'eng', 1: 'not-eng'}
print(f"label2id mapping: {label2id}")
print(f"id2label mapping: {id2label}")
### making output save folders 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_run_name = f"xVector-2sec_Training_{timestamp}"
save_model_path = f"xVector-2sec-saved-model-{timestamp}"
root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn"
save_model_path = os.path.join(root,save_model_path)
chkpt_path = f"{save_model_path}/chkpt"
pth_path = f"{save_model_path}/pthFiles"
eval_path = f"{save_model_path}/evaluations"
# Create the folder if it doesn't exist
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
    os.makedirs(chkpt_path)
    os.makedirs(pth_path)
    os.makedirs(eval_path)
    logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{save_model_path}'.")

train_path  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-pretrained-finetunedONdev-2lang-2sec-HiddenFeatures_full_fast"
result_path = '/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/tdnn'

######################## X_vector ####################
model = X_vector(IP_dim, nc).to(device)
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
            logging.error("Invalid shape of hiddn features (need to be 1024) csv, skipping")
            return None, None
        Xdata1 = []
        f1 = os.path.split(f)[1]     
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

logging.info(f"The custom train dataset has len {len(train_dataset)} and elemnt at index 0 \n{train_dataset[10]}")
logging.info(f"The custom val dataset has len {len(val_dataset)} and elemnt at index 0 \n{val_dataset[10]}")

train_dataloader = DataLoader(  train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

val_dataloader = DataLoader(  val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False)

logging.info(f"Trying out the dataloaders now..")
# Iterate through training DataLoader
print("Training DataLoader:")
for i, (data, label) in enumerate(train_dataloader):
    # Print batch shape
    print(f"Batch {i+1} data shape:", data.shape)
    print(f"Batch {i+1} label shape:", label.shape)

    # Limit the loop to few iterations
    if i >= 2:
        break

# Iterate through validation DataLoader
print("\nValidation DataLoader:")
for i, (data, label) in enumerate(val_dataloader):
    # Print batch shape
    print(f"Batch {i+1} data shape:", data.shape)
    print(f"Batch {i+1} label shape:", label.shape)

    # Limit the loop to few iterations
    if i >= 2:
        break



# #### Actual training
# # p = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/xVectorResults/modelEpoch0_xVector.pth"
# # model.load_state_dict(torch.load(p), strict=False)
# # print(f"Saved Model loaded from {p}")

# if "model_xVector.pth" in os.listdir(root):
#     model.load_state_dict(torch.load(os.path.join(root,"model_xVector.pth")), strict=False)
# else:
#     ## logging into the huggingface to push to the hub and wandb
#     ## loading env variables
#     load_dotenv()
#     secret_value_1 = os.getenv("wandb")

#     if secret_value_1 is None:
#         logging.error(f"Please set Environment Variables properly for wandb login. Exiting.")
#         sys.exit(1)
#     else:
#         # Initialize Wandb with your API keywandb
#         wandb.login(key=secret_value_1)
#         wandb.init(name = wandb_run_name, project="huggingface")
#         logging.info("Login to wandb succesfull!")

#     # Define n_epoch and initialize lists for tracking loss and accuracy
#     train_loss_list = []
#     validation_loss_list = []
#     train_accuracy_list = []
#     validation_accuracy_list = []

#     for e in range(n_epoch):
#         i = 0
#         train_cost = 0.0
#         validation_cost = 0.0
#         random.shuffle(train_df)
#         full_preds = []
#         full_gts = []

#         for s in tqdm(range(0, len(train_df), batch_size)):
#             try: 
#                 XX1, YY1 = getBatch(s,False)
#                 i = i + list(YY1.size())[0]  # Move this line here to count the number of processed files inside the loop
#                 # print(XX1.shape, " --- input shapes--- ",YY1.shape)
#                 # print(YY1)
#                 # print("Iteration/Evaluated so far: ",i)
#             except :
#                 print("Unable to read file, ",f,Exception)
#                 continue
#             if XX1 is None or YY1 is None:
#                 continue
#             # XX1 = torch.unsqueeze(XX1, 1)
#             # X1 = np.swapaxes(XX1, 0, 1)
#             X1 = XX1
#             X1 = Variable(X1, requires_grad=False).to(device)
#             Y1 = Variable(YY1, requires_grad=False).to(device)
#             # print("Target shape: ",YY1.shape, " and ", Y1.shape)
            
#             model.zero_grad()
#             lang_op = model.forward(X1)
#             # print(f'shape of predictions {lang_op.shape} and targets shape is: {Y1.shape}')
#             T_err = loss_lang(lang_op, Y1)
#             T_err.backward()
#             optimizer.step()
#             train_cost = train_cost + T_err.item()
#             # print(i)

#             # Track predictions and ground truth labels
#             predictions = np.argmax(lang_op.detach().cpu().numpy(), axis=1)
#             # print("model out: ",predictions)
#             for pred in predictions:
#                 full_preds.append(pred)
#             for lab in Y1.detach().cpu().numpy():
#                 full_gts.append(lab)
        
#         if i==0:
#             print("Barely Escaped Divison by zero error, skipping this epoch for a good cause...")
#             continue
#         # Calculate and append the average loss for this epoch
#         train_loss_list.append(round(train_cost / i, 4))
#         # Calculate training accuracy for this epoch
#         correct_train_predictions = sum(1 for pred, gt in zip(full_preds, full_gts) if pred == gt)
#         train_accuracy = correct_train_predictions / len(full_gts)
#         train_accuracy_list.append(train_accuracy)

#         # Validation after each epoch
#         validation_full_preds = []
#         validation_full_gts = []

#         for s in tqdm(range(0, len(val_df), batch_size)):
#             try: 
#                 XX_val, YY_val = getBatch(s, True)
#             except:
#                 continue
#             if XX_val is None or YY_val is None:
#                 continue
#             # XX_val = torch.unsqueeze(XX_val, 1)
#             # X_val = np.swapaxes(XX_val, 0, 1)
#             X_val = XX_val
#             X_val = Variable(X_val, requires_grad=False).to(device)
#             Y_val = Variable(YY_val, requires_grad=False).to(device)

#             model.eval()  # Set the model to evaluation mode
#             val_lang_op = model.forward(X_val)
#             val_T_err = loss_lang(val_lang_op, Y_val)
#             validation_cost = validation_cost + val_T_err.item()

#             # Track validation predictions and ground truth labels
#             val_predictions = np.argmax(val_lang_op.detach().cpu().numpy(), axis=1)
#             for pred in val_predictions:
#                 validation_full_preds.append(pred)
#             for lab in Y_val.detach().cpu().numpy():
#                 validation_full_gts.append(lab)

#         # Calculate and append the average validation loss for this epoch
#         validation_loss_list.append(round(validation_cost / len(val_df), 4))

#         # Calculate validation accuracy for this epoch
#         correct_validation_predictions = sum(1 for pred, gt in zip(validation_full_preds, validation_full_gts) if pred == gt)
#         validation_accuracy = correct_validation_predictions / len(validation_full_gts)
#         validation_accuracy_list.append(validation_accuracy)
#         # Log metrics to WandB for this epoch
#         wandb.log({
#             "train_loss": train_loss_list[-1],
#             "train_accuracy": train_accuracy,
#             "validation_loss": validation_loss_list[-1],
#             "validation_accuracy": validation_accuracy
#         })

#         try:
#             result = classification_report(validation_full_preds, validation_full_gts, target_names=label_names)
#             # print(result)
#             # Additional information to include with the report
#             filename_chkpt = f"eval_epoch{e}.txt"
#             # Save the report with additional information to a text file
#             with open(os.path.join(eval_path,filename_chkpt), 'w') as f:
#                 f.write(result)
#             logging.info(f"Evaluated metrics saved at {os.path.join(eval_path,filename_chkpt)}")
#         except:
#             print("Error in evaluate metric compute: ",Exception)

#         # Print epoch, training loss, validation loss, training accuracy, and validation accuracy
#         print(f"Epoch {e + 1}/{n_epoch}: Train Loss={train_loss_list[-1]}, Validation Loss={validation_loss_list[-1]}, Train Accuracy={train_accuracy:.4f}, Validation Accuracy={validation_accuracy:.4f}")

#         #####################################################
#         #### Saving the results and model of each epoch    
#         modelTempName =  f"modelEpoch{e%25}_xVector.pth"
#         torch.save(model.state_dict(),os.path.join(pth_path, modelTempName)) # saving the model parameters 
#         logging.info(f"Checkpointed model at {os.path.join(pth_path,modelTempName)}")
#         ###############################################################

#     # artifact = wandb.Artifact('model', type='model')
#     # artifact.add_file(model_path)
#     # run.log_artifact(artifact)
#     # run.finish()
#     logging.info(f"Xvector training for chunksize {chunk_size} is done!")