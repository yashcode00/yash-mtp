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
import sklearn.metrics
from sklearn.metrics import accuracy_score

##################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# # device_num = torch.cuda.current_device()
# torch.cuda.device(0)
print(torch.cuda.get_device_name(0), torch.cuda.is_available())
# print(torch.cuda.is_available())
##################################################################################
lang2id = {'L1':0, 'L2':1}
id2lang = {0:'L1', 1:'L2'}

train_path  = '/home/dileep/DISPLACE_Challenge/data/original/train/BNF_Feature/'
validation_path  = '/home/dileep/DISPLACE_Challenge/data/original/test/BNF_Feature/'
result_path = '/home/dileep/DISPLACE_Challenge/results/'
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
        self.lstm1 = nn.LSTM(80, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 128, bidirectional=True)
        #self.linear = nn.Linear(2*64,e_dim)
               
        self.fc_ha=nn.Linear(e_dim, 100) 
        self.fc_1= nn.Linear(100, 1)           
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

######################## U_vector ####################
model1 = LSTMNet()
model2 = LSTMNet()

model1.cuda()
model2.cuda()

model = CCSL_Net(model1, model2)
model.cuda()
# optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum= 0.9)
# optimizer_1 = optim.SGD(model.parameters(), lr = 0.001, momentum= 0.9)

# manual_seed = random.randint(1,10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
#######################################################

#### Get the list of all csv files ####
validation_files_list= []
for lang in lang2id.keys():
    for root, dirs, files in os.walk(validation_path + lang):
        for file in files:
            if file.endswith((".csv")): # The arg can be a tuple of suffixes to look for
                validation_files_list.append(os.path.join(root, file))
#### Total number of training files
validation_len = len(validation_files_list)
print('Total Validation files: {}\n'.format(validation_len))

########################################
X1_All, X2_All, Y_All = [], [], []
for fn1 in validation_files_list:    
    tempX1, tempX2, tempY, _ = lstm_data(fn1, train=False)
    X1_All.append(tempX1)
    X2_All.append(tempX2)
    Y_All.append(tempY)
    #print("shape of xx1",XX1.shape)
del tempX1, tempX2, tempY
#######################################

###############################################################################
  
##############################################################################
##### File path to save the test results #####
txtfl = open(result_path + 'u-vector/den_base_validation.txt', 'a')
txtfl.write("\n--- Lookback1={}, Lookback2={}, e_dim={}x2, total_epoch={} ---\n".format(look_back1, look_back2, int(e_dim/2), n_epoch))
# for epoch in range(29, n_epoch):
for epoch in [90, 92, 94, 96, 98]:
    Tru, Pred = [], []
    i = 0
    for fn1 in validation_files_list:
        # X1, X2, Y, _ = lstm_data(fn1, train=False)
        X1, X2, Y = X1_All[i], X2_All[i], Y_All[i]
        i += 1

        X1 = np.swapaxes(X1,0,1)
        X2 = np.swapaxes(X2,0,1)
        x1 = Variable(X1, requires_grad=False).cuda()
        x2 = Variable(X2, requires_grad=False).cuda()
        y = Variable(Y, requires_grad=False).cuda()

        # path = result_path + "u-vector/den_base_model/den_base_noVAD_e"+str(epoch+1)+".pth"
        path = "{}u-vector/den_base_model/den_base_noVAD_{}_{}_e{}.pth".format(result_path, look_back1, look_back2, e+1)
        model.load_state_dict(torch.load(path))
        o1 = model.forward(x1,x2)

        predictions = np.argmax(o1.detach().cpu().numpy(), axis=1)
        for p in predictions:
            Pred.append(p)
        for lab in y.detach().cpu().numpy():
            Tru.append(lab)

    txtfl.write("******* For epoch = {} ********\n".format(epoch+1))
    CM2 = sklearn.metrics.confusion_matrix(Tru, Pred)
    print("Confusion Matrix:\n")
    print(CM2)
    txtfl.write("Confusion Matrix:\n")
    txtfl.write(str(CM2))
    txtfl.write('\n')
    acc = sklearn.metrics.accuracy_score(Tru, Pred)
    print("\nValidation Accuracy: {}".format(acc))
    txtfl.write("Accuracy: " + str(acc))
    txtfl.write('\n')
    txtfl.write("*****************************\n")

txtfl.close()    
###############################################################
################################################################################

