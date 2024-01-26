#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################


################## used Library  ############################################################
import torch
import torch.nn as nn
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
from sklearn.metrics import classification_report, accuracy_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device is: ",device)

############ number of class and all #####################
nc = 11 # Number of language classes 
n_epoch = 100 # Number of epochs
look_back1 = 21 # range
IP_dim = 1024*look_back1 # number of input dimension
# path = "/Users/yash/Desktop/MTP-2k23-24"
path = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/"
model_path = os.path.join(path,"model_xVector.pth")
trigger_times = 0
patience = 6
batch_size = 16
label_names = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'tam', 'tel']
##########################################

lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'tam': 9, 'tel': 10}
id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'tam', 10: 'tel'}

train_path  = "/nlsasfs/home/nltm-st/sujitk/yash/datasets/HiddenFeatures_full_fast"
result_path = '/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/xVectorResults/checkpointsBatched'

######################## X_vector ####################
model = X_vector(IP_dim, nc).to(device)
# model.cuda()

optimizer =  optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output

#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)
#######################################################


#### Function to return data (vector) and target label of a csv (MFCC features) file
def lstm_data(f):
    df = pd.read_csv(f)
    dt = df.astype(np.float32)
    X = np.array(dt).reshape(-1,1024)
    # print("Input shape: ",X.shape) ### (49,1024)
    if X.shape[1] != 1024 or X.shape[0] != 49:
        print("Invalid shape, skipping")
        return None, None
    Xdata1 = []
    f1 = os.path.split(f)[1]     
    lang = f1.split('_')[0] 
    ### target label (output)
    Y1 = np.array([lang2id[lang]])

    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a = X[i:(i+look_back1),:]  
        b = [k for l in a for k in l]      #unpacking nested list(list of list) to list
        Xdata1.append(b)
    Xdata1 = np.array(Xdata1)    
    Xdata1 = torch.from_numpy(Xdata1).float() 
    Y1 = torch.from_numpy(Y1).long()
    # print(Y1.size())
    return Xdata1, Y1

def readInput(s, isVal : bool= False):
    x = []
    y = []
    if isVal:
        data = val_df[s:s+batch_size]
    else:
        data = train_df[s:s+batch_size]

    for file in data:
        try:
            a , b= lstm_data(file)
            if a is None: 
                continue
        except:
            print("Erroe loadinig the file: ",Exception)
            continue
        x.append(a)
        y.append(b)
    x = torch.from_numpy(np.array(x))
    y = torch.from_numpy(np.array(y))
    return x,y


#### Get the list of all csv files ####
files_list= []
for lang in lang2id.keys():
    for f in glob.glob(os.path.join(train_path,lang) + '/*.csv'):
        files_list.append(f)
#### Total number of training files
l = len(files_list)
print('Total Training files: {}\n'.format(l))

# Split the files_list into training and validation sets
train_df , val_df = train_test_split(files_list, test_size=0.2, random_state=42)
print("Training Set: ", len(train_df))
print("Validation Set: ", len(val_df))

train_df = train_df[:10]
val_df = val_df[:10]

#### Actual training

if "model_xVector.pth" in os.listdir(path):
    model.load_state_dict(torch.load(model_path), strict=False)
else:
    ### connecting to wandb
    # Initialize Wandb with your API key
    wandb.login(key="690a936311e63ff7c923d0a2992105f537cd7c59")
    run = wandb.init(name = "LastHopeXVectorTraining", project="huggingface")

    # Define n_epoch and initialize lists for tracking loss and accuracy
    train_loss_list = []
    validation_loss_list = []
    train_accuracy_list = []
    validation_accuracy_list = []

    for e in range(n_epoch):
        i = 0
        train_cost = 0.0
        validation_cost = 0.0
        random.shuffle(train_df)
        full_preds = []
        full_gts = []

        for fn in train_df:
            try: 
                XX1, YY1 = lstm_data(fn)
                i = i + 1  # Move this line here to count the number of processed files inside the loop
                # print(i)
            except :
                print("Unable to read file, ",f,Exception)
                continue
            if XX1 is None or YY1 is None:
                continue
            XX1 = torch.unsqueeze(XX1, 1)
            X1 = np.swapaxes(XX1, 0, 1)
            print("INput shape: ",X1.shape)
            X1 = Variable(X1, requires_grad=False).to(device)
            Y1 = Variable(YY1, requires_grad=False).to(device)
            
            model.zero_grad()
            lang_op = model.forward(X1)
            T_err = loss_lang(lang_op, Y1)
            T_err.backward()
            optimizer.step()
            train_cost = train_cost + T_err.item()
            # print(i)

            # Track predictions and ground truth labels
            predictions = np.argmax(lang_op.detach().cpu().numpy(), axis=1)
            print("model out: ",predictions)
            for pred in predictions:
                full_preds.append(pred)
            for lab in Y1.detach().cpu().numpy():
                full_gts.append(lab)
        
        if i==0:
            print("Barely Escaped Divison by zero error, skipping this epoch of a good cause...")
            continue
        # Calculate and append the average loss for this epoch
        train_loss_list.append(round(train_cost / i, 4))
        # Calculate training accuracy for this epoch
        correct_train_predictions = sum(1 for pred, gt in zip(full_preds, full_gts) if pred == gt)
        train_accuracy = correct_train_predictions / len(full_gts)
        train_accuracy_list.append(train_accuracy)

        # Validation after each epoch
        validation_full_preds = []
        validation_full_gts = []

        for val_fn in val_df:
            try: 
                XX_val, YY_val = lstm_data(val_fn)
            except:
                continue
            if XX_val is None or YY_val is None:
                continue
            XX_val = torch.unsqueeze(XX_val, 1)
            X_val = np.swapaxes(XX_val, 0, 1)
            X_val = Variable(X_val, requires_grad=False)
            Y_val = Variable(YY_val, requires_grad=False)

            model.eval()  # Set the model to evaluation mode
            val_lang_op = model.forward(X_val)
            val_T_err = loss_lang(val_lang_op, Y_val)
            validation_cost = validation_cost + val_T_err.item()

            # Track validation predictions and ground truth labels
            val_predictions = np.argmax(val_lang_op.detach().cpu().numpy(), axis=1)
            for pred in val_predictions:
                validation_full_preds.append(pred)
            for lab in Y_val.detach().cpu().numpy():
                validation_full_gts.append(lab)

        # Calculate and append the average validation loss for this epoch
        validation_loss_list.append(round(validation_cost / len(val_df), 4))

        # Calculate validation accuracy for this epoch
        correct_validation_predictions = sum(1 for pred, gt in zip(validation_full_preds, validation_full_gts) if pred == gt)
        validation_accuracy = correct_validation_predictions / len(validation_full_gts)
        validation_accuracy_list.append(validation_accuracy)
        # Log metrics to WandB for this epoch
        wandb.log({
            "train_loss": train_loss_list[-1],
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss_list[-1],
            "validation_accuracy": validation_accuracy
        })

        try:
            result = classification_report(validation_full_preds, validation_full_gts, target_names=label_names)
            # print(result)
            # Additional information to include with the report
            additional_info = f"/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/xVectorResults/evaluations/eval_epoch{e}.txt"
            # Save the report with additional information to a text file
            with open(additional_info, 'w') as f:
                f.write(result)
        except:
            print("Error in evaluate metric compute: ",Exception)

        # Print epoch, training loss, validation loss, training accuracy, and validation accuracy
        print(f"Epoch {e + 1}/{n_epoch}: Train Loss={train_loss_list[-1]}, Validation Loss={validation_loss_list[-1]}, Train Accuracy={train_accuracy:.4f}, Validation Accuracy={validation_accuracy:.4f}")

        #####################################################
        #### Saving the results and model of each epoch    
        torch.save(model.state_dict(),os.path.join(result_path, f"modelEpoch{e%50}_xVector.pth")) # saving the model parameters 
        print("Checkpointed.. ")
        ###############################################################


        ### Early stopping
        # if len(validation_loss_list)>1 and validation_loss_list[-1] > validation_loss_list[-2]:
        #     trigger_times += 1
        #     print('Trigger Times:', trigger_times)

        #     if trigger_times >= patience:
        #         print('Early stopping!')
        #         break
        # else:
        #     print('trigger times: 0')
        #     trigger_times = 0
        
        # if len(train_loss_list) >1 and  abs(train_loss_list[-1] - train_loss_list[-2]) < 1e5:
        #     print("Early stopping")
        #     break

    # artifact = wandb.Artifact('model', type='model')
    # artifact.add_file(model_path)
    # run.log_artifact(artifact)
    # run.finish()
    print("Training Complete and model is saved at: ", os.path.join(result_path, f"modelEpoch{n_epoch}_xVector.pth"))