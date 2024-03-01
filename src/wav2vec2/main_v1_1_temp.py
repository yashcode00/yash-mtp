## author @Yash Sharma, B20241
## imporying all the neccesary modules
## loading important libraries
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
from transformers import EarlyStoppingCallback
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import evaluate
import time
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import logging 
from dotenv import load_dotenv

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# disable_caching()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ",device)
if device=='cuda':
    torch.cuda.empty_cache()

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

base_directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/"
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_offline_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2/displce-2sec-finetunedOndev-300M-saved-model_20240218_143551/pthFiles/model_epoch_9"
processor_tokenizer_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_run_name = f"displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_{timestamp}"
save_model_path = f"displace-terminator-pretrained-finetune-onDev-rttm-300M-saved-model_{timestamp}"
save_model_path = os.path.join("/nlsasfs/home/nltm-st/sujitk/yash-mtp/models/wav2vec2",save_model_path)
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
cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/cache"
dataset_path= "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-fromRttm-saved-dataset.hf"

# We need to specify the input and output column
input_column = "path"
output_column = "language"
label_names = ['eng', 'not-eng']
model_out_dir = os.path.join(cache_dir, "wav2vec2-large-xls-r-300m-indian-language-classification-displace")

num_epochs = 50
## this batch size is not used , please refer to batch size per device in training args as this 
## is a distributed training.
batch_size = 256  ## not used 
proxy_url = "http://proxy-10g.10g.siddhi.param:9090"
chunk_size = 32000 ## the audio chunk size that is used for finetuinng like in this case a  :i.e. 32000 -> 2 sec chunks samplec at 16kHz

##################################################################################################
##################################################################################################

## loading from saved dataset
dataset = load_from_disk(dataset_path)
logging.info("Datasets loaded succesfully!")

# dataset = concatenate_datasets([dataset["train"],dataset["test"],dataset["validation"]])

logging.info("Loaded the following dataset: \n",dataset)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
print("Train: ",train_dataset)
print("Validation: ",eval_dataset)

print("After concatenation: ")
train_dataset = concatenate_datasets([dataset["train"] ,dataset["validation"]])
print(f"train dataset is :{train_dataset}")

# train test split
# temp = train_dataset.train_test_split(test_size=0.5)
# train_dataset = temp["test"]
# temp = eval_dataset.train_test_split(test_size=0.5)
# eval_dataset = temp["test"]
# dataset.cleanup_cache_files()

# train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]
# logging.info("After: ")
# logging.info("Train: ",train_dataset)
# logging.info("Validation: ",eval_dataset)
# logging.info("Test: ",test_dataset)
## defining input and output columns

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
logging.info(f"A classification problem with {str(num_labels)} classes: {str(label_list)}")


###############################################################################################################
### Now fetching all data and componenets of the specified model
###############################################################################################################
## loading the wave2vec2 feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_url , cache_dir=cache_dir)
### loading the processor and tokenizer contained inside it
pooling_mode = "mean"
# config
config = AutoConfig.from_pretrained(
    repo_url,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    cache_dir=cache_dir,
)
setattr(config, 'pooling_mode', pooling_mode)
## Loading the processor for wav2vec2
processor = Wav2Vec2Processor.from_pretrained(processor_tokenizer_url, cache_dir=cache_dir)
## loading the main model
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_offline_path,
    config=config , 
    cache_dir=cache_dir
)

## for transfer learning
model.freeze_feature_extractor()

#### Done loading model and its components ####################################################################
###############################################################################################################

# Preprocess
target_sampling_rate = feature_extractor.sampling_rate

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    inputs = feature_extractor(
        speech_list, sampling_rate=feature_extractor.sampling_rate, max_length=chunk_size, truncation=True
    )
    inputs["labels"] = list(target_list)
    return inputs


torch.set_num_threads(1)  ## imp

train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=1024,
    batched=True,
    num_proc=500,
    # keep_in_memory=True
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=1024,
    batched=True,
    num_proc=500,
    # keep_in_memory=True
)

logging.info(f"The final processed dataset is as below: ")
print(train_dataset)
print(f"Size of input values shall be {chunk_size} smaples which is here : {len(train_dataset[0]['input_values'])}")
    
label2id={label: i for i, label in enumerate(label_list)}
id2label={i: label for i, label in enumerate(label_list)}
logging.info(f"label_names : {str(label_names)}")

target_sampling_rate = processor.feature_extractor.sampling_rate
logging.info(f"The target sampling rate: {target_sampling_rate}")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
is_regression = False


eval_steps = int(len(train_dataset)/batch_size)
print("Eval steps: ",str(eval_steps))

fp16 = True
training_args = TrainingArguments(
    output_dir=model_out_dir,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    overwrite_output_dir=True,
    num_train_epochs=250,
    save_steps=1*eval_steps,  # Save checkpoints every 1000 steps
    logging_steps=200,
    learning_rate=5e-5,
    save_total_limit=5,
    # fp16=fp16,
    run_name="resampled_data_training_500_8Gpu",
    load_best_model_at_end=True,  # Load the best model checkpoint at the end
    # callbacks=[early_stopping_callback],  # Add the callbacks
    metric_for_best_model="eval_accuracy",  # Metric to determine the best model
    greater_is_better=True,  # Set to True if higher metric values are better
    save_strategy="steps",  # Save strategy
)



##############################################################################################################################
### Main Distributed Training Part
##############################################################################################################################
logging.info("-*-"*100)
logging.info("The Training is about to start....")

#### Using accelerate to train over multiple gpus
### making data loaders

dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size)
accelerator = Accelerator(mixed_precision= 'fp16')

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
    wandb.init(name = wandb_run_name, project="huggingface")

 # Instantiate dataloaders.
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    collate_fn=data_collator, 
    batch_size=training_args.per_device_train_batch_size,
    drop_last=True
)
eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=training_args.per_device_eval_batch_size,
    drop_last=(Accelerator.mixed_precision == "fp8"),
)

# If the batch size is too big we use gradient accumulation
gradient_accumulation_steps = 1
model = model.to(accelerator.device)
print("Device of accleration: ",str(accelerator.device))
# Instantiate optimizer
optimizer = AdamW(params=model.parameters(), lr=3e-5)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=(len(train_dataloader) * num_epochs) ,
)

# Prepare everything
# There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
# prepare method.
if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

try: 
    # Register the LR scheduler
    accelerator.register_for_checkpointing(lr_scheduler)
    # Save the starting state
    accelerator.save_state(chkpt_path)
    logging.info("Started checkpointing")
except Exception as err:
    logging.info("Can't checkpoint... error: ",err)


num_training_steps = num_epochs * len(train_dataloader)
i=0
shallPrintGpuUsage = 0
progress_bar = tqdm(range(num_training_steps))
# Now we train the model
for epoch in range(num_epochs):
    train_loss = []
    val_loss = []
    final_train_loss = -1
    final_val_loss = -1
    model.train()
    for step, batch in (enumerate(train_dataloader)):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        outputs = model(**batch)
        loss = outputs.loss
        # loss = loss / gradient_accumulation_steps
        train_loss.append(accelerator.gather(loss))
        accelerator.backward(loss)
        progress_bar.update(1)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if shallPrintGpuUsage%100 ==0:
            print_gpu_info()
        shallPrintGpuUsage +=1 
    final_train_loss = sum(train_loss)/len(train_loss)

    ### chekpointing the model
    try:
        # Save the starting state
        logging.info("Saving Model..")
        accelerator.save_state(chkpt_path)
        # Save the model after every epoch
        model.save_pretrained(os.path.join(pth_path,f"model_epoch_{epoch%6}"))
    except Exception as err:
        logging.error("Some exception occured while saving the model: ",err)

    
    logging.info("Evaluating Wait...")
    x = np.array([])
    y = np.array([])
    model.eval()
    for step, batch in (enumerate(eval_dataloader)):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        loss = outputs.loss
        val_loss.append(accelerator.gather(loss))
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        try:
            x = np.concatenate((x,predictions.cpu().numpy()),axis=0)
            y = np.concatenate((y,references.cpu().numpy()),axis=0)
        except Exception as err:
            logging.error("Error Converting to np and processing the x and y: ",err)
        if i==0:
            logging.info("Shape of each eval prediction: " ,str(predictions.shape))
            i =1
    final_val_loss = sum(val_loss)/len(val_loss)


    try:
        result = classification_report(x, y, target_names=label_names)
        # logging.info(result)
        # Additional information to include with the report
        additional_info = os.path.join(eval_path,f"eval_epoch{epoch}.txt")
        # Save the report with additional information to a text file
        with open(additional_info, 'w') as f:
            f.write(result)
    except:
        logging.error("Error in evaluate metric compute: ",str(Exception))
    print("Shape of predictions: ",x.shape,"--",x[:4])
    print("Shape of targets: ",y.shape,"--", y[:4])
    accuracy = accuracy_score(x,y)
    # Use accelerator.logging.info to logging.info only on the main process.
    try:
        # Log metrics to WandB for this epoch
        wandb.log({
            "validation_accuracy": accuracy,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
        })
    except Exception as err:
        logging.error("Not able to log to wandb, ", err)

    accelerator.print(f"Epoch {epoch+1}/{num_epochs}: train_loss: {final_train_loss} val_loss: {final_val_loss} Val_Accuracy:{accuracy}")

logging.info("\nTraining Done!")
logging.info("$$"*100)
logging.info("Work done mate")

# # trainer = Trainer(
# #     model=model,
# #     data_collator=data_collator,
# #     args=training_args,
# #     compute_metrics=compute_metrics,
# #     train_dataset=train_dataset,
# #     eval_dataset=eval_dataset,
# #     tokenizer=processor.feature_extractor,
# # )

# # train_result = trainer.train()
# ## finding the metrics
# # After the training/evaluation, you can use the following code to save metrics.
# # Compute train results
# metrics = train_result.metrics
# max_train_samples = len(train_dataset)
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))

# # Save train results
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)

# # Compute evaluation results
# metrics = trainer.evaluate()
# max_val_samples = len(eval_dataset)
# metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

# # Save evaluation results
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)

# # Save logs to the output directory
# trainer.save_state()

# ## logging into the huggingface to push to the hub
# secret_value_0 = "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"
# login(secret_value_0)


# ### setting up the proxy
# try: 
#     # Set HTTP proxy
#     os.environ["http_proxy"] = proxy_url
#     os.environ["HTTP_PROXY"] = proxy_url

#     # Set HTTPS proxy
#     os.environ["https_proxy"] = proxy_url
#     os.environ["HTTPS_PROXY"] = proxy_url

#     # Set FTP proxy
#     os.environ["ftp_proxy"] = proxy_url
#     os.environ["FTP_PROXY"] = proxy_url
#     # Check if the proxy variables are set
#     logging.info("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
#     logging.info("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
#     logging.info("FTP_PROXY:", os.environ.get("FTP_PROXY"))
# except Exception as err:
#     logging.info("Error setting up the proxies: ", err)

# ####


# pathx = f"final_model_saved"
# trainer.save_model(pathx)
# logging.info("Model saved at : ",pathx," cheers!!")

# logging.info(f"Pushing to hub, you can find at {repo_url}")
# trainer.push_to_hub(repo_url)
# processor.push_to_hub(repo_url)

# logging.info("Work done mate")
