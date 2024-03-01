# WSSL code fine-tuning
# MKH 28-06-2021
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np
import pandas as pd

import glob
import random

from torch.autograd import Variable
from torch.autograd import Function
from torch import optim

import sklearn.metrics
from sklearn.metrics import accuracy_score

##################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# # device_num = torch.cuda.current_device()
# torch.cuda.device(0)
print(torch.cuda.get_device_name(0), torch.cuda.is_available())
# print(torch.cuda.is_available())
##################################################################################
lang2id = {'asm':0, 'ben':1, 'eng':2, 'guj':3, 'hin':4, 'kan':5, 'mal':6, 'mar':7, 'odi':8, 'pun':9, 'tam':10, 'tel':11}
id2lang = {0:'asm', 1:'ben', 2:'eng', 3:'guj', 4:'hin', 5:'kan', 6:'mal', 7:'mar', 8:'odi', 9:'pun', 10:'tam', 11:'tel'}

##### Modifying e1e2inter2aa
Nc = 12 # Number of language classes 
n_epoch = 50 # Number of epochs

look_back1 = 20     # Chunk size for embedding extractor-1
look_back2 = 50     # Chunk size for embd extractor-2

def lstm_data(fname, label, train=True):
    df = pd.read_csv(fname, encoding='utf-16', usecols=list(range(0,80)))
    # print(fname, df.shape)
    dt = df.astype(np.float32)
    # if dt.shape[0] <= look_back2:
    #     mul_len = int((look_back2 + 1)/dt.shape[0]) + 1
    #     # dt = dt.iloc[np.tile(np.arange(dt.shape[0]), mul_len)]
    #     dt = pd.concat([dt]*mul_len, ignore_index=True)
    X = np.array(dt)
    
    Xdata1 = []
    Xdata2 = [] 
    Ydata1 = []
      
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std 
    # # f1 = os.path.splitext(f)[0]     
        
    # #### Change when target level (value) changed
    # if train:
    #     le=len(train_path)
    # else:
    #     le=len(test_path)
    # lang = f[le:le+3]
    # # lang = f1[63:66]   

    ### target label (output)
    Y1 = lang2id[label]
    Y2 = np.array([Y1])
    
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,2):     #Low resolution long context       
        b=X[i+1:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)
    
    # print(Xdata1.shape, Xdata2.shape)

    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
    Ydata1 = torch.from_numpy(Y2).long()    
    
    return Xdata1,Xdata2,Ydata1,Y1

###########################
look_back1=20 # Chunk size for embedding extractor-1
look_back2=50 # Chunk size for embd extractor-2

#################################################################################### Modifying e1e2inter2aa
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 32,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*32,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alp= self.fc_1(ha)
        al= self.sftmax(alp) 
        
        T=list(ht.shape)[1]  
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))  # Self-attention on LID-seq-senones to get utterance-level embedding (e1/e2)      
        c = torch.squeeze(c,0)        
        return (c)

class MSA_DAT_Net(nn.Module):
    def __init__(self, model1,model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1=nn.Linear(2*32,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(2*32, Nc, bias=True))
        
    def forward(self, x1,x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)        
        ht_u = torch.cat((u1,u2), dim=0)  
        ht_u = torch.unsqueeze(ht_u, 0) 
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alp)
        Tb = list(ht_u.shape)[1] 
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D)) # Self-attention combination of e1 and e2 to get u-vec
        u_vec = torch.squeeze(u_vec,0)
        
        lang_output = self.lang_classifier(u_vec)      # Output layer  
        
        return (lang_output,u1,u2,u_vec)
###############################################################################################

##############################################################################
# manual_seed = random.randint(1,10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
##############################################################################
# files_list=[]
# folders = glob.glob('/home/iit/disk_10TB/Muralikrishna/wssl_wav/train_wav_2_bnf/*')  # Train files in csv format
# for folder in folders:
#     for f in glob.glob(folder+'/*.csv'):
#         files_list.append(f)
             
# l = len(files_list)
# random.shuffle(files_list)
# print('Total Training files: ',l)

##############################################################################
df1 = pd.read_csv('test_file_list_from_iHub_Data_BNF_Feature_Data_Final.csv', header=None)
test_files_list = df1[0].values
test_labels = df1[1].values
#### Total number of testing files
test_len = len(test_files_list)
print('Total Test files: {}\n'.format(test_len))
##############################################################################

##### File path to save the testing results #####
result_path = '/home/dileep/Data1/sujeet/results/uVector_wssl/'
os.makedirs(result_path + 'model/', exist_ok=True)
##############################################################################

print(" ########################################################################################################  ")

for e in range(n_epoch):
    
    model1 = LSTMNet()
    model2 = LSTMNet()

    model1.cuda()
    model2.cuda()

    model = MSA_DAT_Net(model1, model2)

    ### For loading a save model for retraining
    model_path = "{}/model/ZWSSL_{}_{}_e{}.pth".format(result_path, look_back1, look_back2, e+1)
    if not os.path.isfile(model_path):
        break

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    i=0  # number of files completed in the epoch
    full_preds=[]
    full_gts=[]
    txtfl = open(result_path + 'test_ZWSSL.txt', 'a')

    for fn in test_files_list:    
        ### Load/Get the data
        XX1,XX2,YY1,Yint = lstm_data(fn, test_labels[i], train=False)
        XNP=np.array(XX1)
        if(np.isnan(np.sum(XNP))):
            continue
        
        XNP=np.array(XX2)
        if(np.isnan(np.sum(XNP))):
            continue
    
        i = i+1
        XX1 = np.swapaxes(XX1,0,1)
        XX2 = np.swapaxes(XX2,0,1)
        X1 = Variable(XX1,requires_grad=False).cuda()
        Y1 = Variable(YY1,requires_grad=False).cuda()
        X2 = Variable(XX2,requires_grad=False).cuda()
        
        fl,_,_,_ = model.forward(X1,X2)       
          
        ### Get the prediction
        predictions = np.argmax(fl.detach().cpu().numpy(), axis=1)
        # print(predictions)
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)
    mean_acc = accuracy_score(full_gts, full_preds)
    CM2 = sklearn.metrics.confusion_matrix(full_gts, full_preds)
    print(CM2)
    txtfl.write(model_path)
    txtfl.write('\n')
    txtfl.write('Total Test Accuracy = {} after {} epochs'.format(mean_acc,e+1))
    txtfl.write('\n')
    txtfl.write(str(CM2))
    txtfl.write('\n')
    # mean_loss = np.mean(np.asarray(T_err.detach().cpu()))
    print('Total Test Accuracy {} after {} epochs'.format(mean_acc,e+1)) 
    txtfl.write('\n')
    txtfl.close()
