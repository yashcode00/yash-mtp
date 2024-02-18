################## used Library  ############################################################
import torch
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/common")
from dotenv import load_dotenv
import os 
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
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
import logging
from datetime import datetime
from huggingface_hub import login


## logging into the huggingface to push to the hub and wandb
## loading env variables
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


givenName = "yashcode00/wave2vec2-1sec-2lang-finetuned-300m-xlsr"

model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-1sec-300M-saved-model-20240213_215511/pthFiles/model_epoch_1/pytorch_model.bin"
# config = AutoConfig.from_pretrained(model_name_or_path)
# processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
# model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(offline_model_path)

run =  wandb.init(name = "model-displace-wave2vec2-1sec", project="huggingface")
artifact = wandb.Artifact('model', type='model')
artifact.add_file(local_path=offline_model_path)
run.log_artifact(artifact)
run.finish()
print("Finished uploading the artifact. 1")
print("Succesfully pushed")


offline_model_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displace-2sec-300M-saved-model-20240214_193154/pthFiles/model_epoch_0/pytorch_model.bin"
run =  wandb.init(name = "model-displace-wave2vec2-2sec", project="huggingface")
artifact = wandb.Artifact('model', type='model')
artifact.add_file(local_path=offline_model_path)
run.log_artifact(artifact)
run.finish()
print("Finished uploading the artifact. 2")
print("Succesfully pushed")
