# Disentangle Network
# MKH --- 30-Nov-2021

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
from sklearn.metrics import accuracy_score

##################################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# # # device_num = torch.cuda.current_device()
# # torch.cuda.device(0)
# print(torch.cuda.get_device_name(0), torch.cuda.is_available())
# # print(torch.cuda.is_available())
##################################################################################
lang2id = {'L1':0, 'L2':1}
id2lang = {0:'L1', 1:'L2'}

train_path  = '/home/sujeet/Data/Sujeet_PhD/Research_Work/paper_implementation/sample_data/MFCC_Features/Train/'
validation_path  = '/home/sujeet/Data/Sujeet_PhD/Research_Work/paper_implementation/sample_data/MFCC_Features/Test/'
result_path = '/home/sujeet/Data/Sujeet_PhD/Research_Work/paper_implementation/sample_data/MFCC_Features/results/'
##################################################################################
##### Modifying e1e2inter2aa
e_dim = 128*2
Nc = 2 # Number of language classes 
n_epoch = 100 # Number of epochs

look_back1 = 4
look_back2 = 8

def lstm_data(f, train=True):
    df = pd.read_csv(f, encoding='utf-16', usecols=list(range(0,80)))
    dt = df.astype(np.float32)
    if dt.shape[0] <= look_back2:
        mul_len = int((look_back2 + 1)/dt.shape[0]) + 1
        # dt = dt.iloc[np.tile(np.arange(dt.shape[0]), mul_len)]
        dt = pd.concat([dt]*mul_len, ignore_index=True)
        
    # if dt.shape[0] <= (look_back2 * 2):
    #     mul_len = int((look_back2 * 2 - 1)/dt.shape[0]) + 1
    #     # dt = dt.iloc[np.tile(np.arange(dt.shape[0]), mul_len)]
    #     dt = pd.concat([dt]*mul_len, ignore_index=True)
        
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
    if train:
        le=len(train_path)
    else:
        le=len(validation_path)
    lang = f[le:le+2]
    # lang = f1[63:66]   

    ### target label (output)
    Y1 = lang2id[lang]
    Y2 = np.array([Y1])
    
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
    Ydata1 = torch.from_numpy(Y2).long()    
    
    return Xdata1,Xdata2,Ydata1,Y1

###############################################################################
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 128,bidirectional=True)
        #self.linear = nn.Linear(2*64,e_dim)
               
        self.fc_ha=nn.Linear(e_dim,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0) 
        #ht = torch.tanh(self.linear(ht))      
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp) 
        
       
        T = list(ht.shape)[1]  
        batch_size = list(ht.shape)[0]
        D = list(ht.shape)[2]
        c = torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))        
        c = torch.squeeze(c,0)        
        return (c)

class CCSL_Net(nn.Module):
    def __init__(self, model1,model2):
        super(CCSL_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        self.att_fc = nn.Linear(e_dim,e_dim)
        #self.cla_fc = nn.Linear(e_dim,e_dim)
        
        self.sftmx = torch.nn.Softmax(dim=1)

        self.lang_classifier = nn.Linear(e_dim, Nc, bias = True)
        self.adv_classifier = nn.Linear(e_dim, Nc, bias = True) 
        
        
    def attention(self, att, cla):

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)
        norm_att = att / torch.sum(att, dim=1)[:, None, :]

        u_LID = torch.sum(norm_att * cla, dim=1)  # Disentagle LID-specific and channel-specific u-vectors
        u_ch = torch.sum(1-norm_att * cla, dim=1)
        
        return u_LID, u_ch   
        
        
    def forward(self, x1,x2):
        e1 = self.model1(x1)
        e2 = self.model2(x2) 
        
        att_input = torch.cat((e1,e2), dim=0)
        att_input = torch.unsqueeze(att_input, 0)
        
        att = torch.sigmoid(self.att_fc(att_input))
        cla = att_input # No additional layer 
        u_lid, u_ch = self.attention(att, cla) # Get LID-specific and channel-specific u-vectors.

        lang_output = self.lang_classifier(u_lid)      # Restitute the u_lid  
        lang_output = self.sftmx(lang_output) # Langue prediction from language classifier
        
        return (lang_output)
        
###############################################################################################

model1 = LSTMNet()
model2 = LSTMNet()

model1.cuda()
model2.cuda()

model = CCSL_Net(model1,model2)
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.001, momentum= 0.9)

loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')
loss_lang.cuda()


target = torch.ones(1,Nc).to("cuda")
print('The target tensor=',target)

manual_seed = random.randint(1,10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################################################

#### Get the list of all csv files ####
train_files_list= []
for lang in lang2id.keys():
    for root, dirs, files in os.walk(train_path + lang):
        for file in files:
            if file.endswith((".csv")): # The arg can be a tuple of suffixes to look for
                train_files_list.append(os.path.join(root, file))

#### Total number of training files
train_len = len(train_files_list)
print('Total Training files: {}\n'.format(train_len))

##### File path to save the training results #####
os.makedirs(result_path + 'u-vector/den_base_model/', exist_ok=True)
txtfl = open(result_path + 'u-vector/den_base_train.txt', 'a')
txtfl.write("\n--- Lookback1={}, Lookback2={}, e_dim={}x2, total_epoch={} ---\n".format(look_back1, look_back2, int(e_dim/2), n_epoch))
#################################################
random.shuffle(train_files_list)
# ########################################
# X1_All, X2_All, Y_All = [], [], []
# for fn in train_files_list:    
#     tempX1, tempX2, tempY, _ = lstm_data(fn)
#     X1_All.append(tempX1)
#     X2_All.append(tempX2)
#     Y_All.append(tempY)
#     #print("shape of xx1",XX1.shape)
# del tempX1, tempX2, tempY
# #######################################
##############################################################################
for e in range(n_epoch):
    i = 0
    cost = 0.0
    full_preds=[]
    full_gts=[]
    for fn in train_files_list:    
        # #print(fn)
        # df = pd.read_csv(fn,encoding='utf-16',usecols=list(range(0,80)))
        # data = df.astype(np.float32)
        # X = np.array(data) 
        # N, D = X.shape

        # if N>look_back2:
        model.zero_grad()
    
        XX1,XX2,YY1,Yint = lstm_data(fn)
        # XX1, XX2, YY1 = X1_All[i], X2_All[i], Y_All[i]

        XNP=np.array(XX1)
        if(np.isnan(np.sum(XNP))):
            continue
        
        XNP=np.array(XX2)
        if(np.isnan(np.sum(XNP))):
            continue
    
        i = i+1
        XX1 = np.swapaxes(XX1,0,1)
        # print(XX2)
        XX2 = np.swapaxes(XX2,0,1)
        X1 = Variable(XX1,requires_grad=False).cuda()
        Y1 = Variable(YY1,requires_grad=False).cuda()
        X2 = Variable(XX2,requires_grad=False).cuda()
        

        lang_op = model.forward(X1,X2)
        err_l = loss_lang(lang_op,Y1)

            
        T_err = err_l          
        T_err.backward()
        
        optimizer.step()
        cost = cost + T_err.item()
        
        print("u-vector_den_base: Epoch = ",e,"  completed files  "+str(i)+"/"+str(train_len)+" Loss= %.5f"%(cost/i))  
        predictions = np.argmax(lang_op.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)
    mean_acc = accuracy_score(full_gts, full_preds)
    mean_loss = np.mean(np.asarray(T_err.detach().cpu()))
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,e+1)) 
    path = "{}u-vector/den_base_model/den_base_noVAD_{}_{}_e{}.pth".format(result_path, look_back1, look_back2, e+1)
    if e >= 90 and (e % 2)==0:
        torch.save(model.state_dict(),path)
    txtfl.write(path)
    txtfl.write('\n')
    txtfl.write(str(mean_acc))
    txtfl.write('\n')

################################################################################

