## author @Yash Sharma, B20241
## imporying all the neccesary modules
## loading important libraries
from tqdm import tqdm
import torchaudio
import os
from datasets import  load_from_disk, concatenate_datasets
from transformers.file_utils import ModelOutput
import torch
from datetime import datetime 
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoFeatureExtractor, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer
# is_apex_available
import torch
from packaging import version
from torch import nn
from huggingface_hub import login
import wandb
from Model import *
torch.cuda.empty_cache()
from transformers import EarlyStoppingCallback
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import evaluate
import time
from sklearn.metrics import classification_report, accuracy_score

# disable_caching()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
print("Device is: ",device)
chkpt_path = "/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Saved_Models2/chkpt"

cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
dir = "/nlsasfs/home/nltm-st/sujitk/yash/datasets"
final_path= os.path.join(dir,"saved_dataset.hf")
save_path = cache_dir
# i = 0
# final_path = f"/nlsasfs/home/nltm-st/sujitk/yash/datasets/disjointChunks/chunk_{i}"

## loading from saved dataset
dataset = load_from_disk(final_path)
print("DAtasets loaded succesfully!!")

# dataset = concatenate_datasets([dataset["train"],dataset["test"],dataset["validation"]])

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
print("Before: ")
print("Train: ",train_dataset)
print("Validation: ",eval_dataset)


# train test split
temp = train_dataset.train_test_split(test_size=0.6)
train_dataset = temp["test"]
temp = eval_dataset.train_test_split(test_size=0.6)
eval_dataset = temp["test"]
# dataset.cleanup_cache_files()

# train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

print("After: ")
print("Train: ",train_dataset)
print("Validation: ",eval_dataset)
# print("Test: ",test_dataset)
## defining input and output columns
# We need to specify the input and output column
input_column = "path"
output_column = "language"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

# Preprocess
# The next step is to load a Wav2Vec2 feature extractor to process the audio signal:
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_url , cache_dir=cache_dir)

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
        speech_list, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    inputs["labels"] = list(target_list)
    return inputs


train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=1024,
    batched=True,
    num_proc=50,
    # keep_in_memory=True
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=1024,
    batched=True,
    num_proc=50,
    # keep_in_memory=True
)

label2id={label: i for i, label in enumerate(label_list)}
id2label={i: label for i, label in enumerate(label_list)}
# label_names = [id2label[0][i] for i in range(num_labels)]
label_names = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'tam', 'tel']
print("label_names",label_names)

### loading the processor and tokenizer contained inside it
pooling_mode = "mean"
# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    cache_dir=cache_dir,
)
setattr(config, 'pooling_mode', pooling_mode)


processor = Wav2Vec2Processor.from_pretrained(repo_url, cache_dir=cache_dir)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
is_regression = False

    
## loading the main models
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
#     num_labels=num_labels,
#     label2id={label: i for i, label in enumerate(label_list)},
#     id2label={i: label for i, label in enumerate(label_list)},
    config=config , 
    cache_dir=cache_dir
)

model.freeze_feature_extractor()

batch_size = 256
eval_steps = int(len(train_dataset)/batch_size)
print("Eval steps: ",eval_steps)
model_out_dir = os.path.join(save_path, "wav2vec2-large-xlsr-indian-language-classification-featureExtractor")
# checkpt = os.path.join(save_path, "chkpt-wav2vec2-large-xlsr-indian-language-classification-featureExtractor")
proxy_url = "http://proxy-10g.10g.siddhi.param:9090"



early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement to wait before stopping
    early_stopping_threshold=0.0,  # Minimum improvement required to consider it as an improvement
)

model.to(device)
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
    fp16=fp16,
    run_name="resampled_data_training_500_8Gpu",
    load_best_model_at_end=True,  # Load the best model checkpoint at the end
    # callbacks=[early_stopping_callback],  # Add the callbacks
    metric_for_best_model="eval_accuracy",  # Metric to determine the best model
    greater_is_better=True,  # Set to True if higher metric values are better
    save_strategy="steps",  # Save strategy
)


##########################################
print("$$"*100)
print("The Training is about to start....")

#### Using accelerate to train over multiple gpus
### making data loaders

dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size)
accelerator = Accelerator(mixed_precision= "fp16" if training_args.fp16 else None)

 # Instantiate dataloaders.
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, drop_last=True
)
eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=training_args.per_device_eval_batch_size,
    drop_last=(Accelerator.mixed_precision == "fp8"),
)

# If the batch size is too big we use gradient accumulation
num_epochs = 500
gradient_accumulation_steps = 1
model = model.to(accelerator.device)
print("Device of acclt: ", accelerator.device)
# Instantiate optimizer
optimizer = AdamW(params=model.parameters(), lr=2e-5)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
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
    print("Started checkpointing")
except Exception as err:
    print("Can't checkpoint... error: ",err)


# ### connecting to wandb
# Initialize Wandb with your API key
wandb.login(key="690a936311e63ff7c923d0a2992105f537cd7c59")
wandb.init(name = "3-secondResampled_Half_data_training_50_8Gpu", project="huggingface")

num_training_steps = num_epochs * len(train_dataloader)
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
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    final_train_loss = sum(train_loss)/len(train_loss)

    ### chekpointing the model
    try:
        # Save the starting state
        print("Saving Mod3l..")
        accelerator.save_state(chkpt_path)
        # Save the model after every epoch
        model.save_pretrained(f"/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Saved_Models2/pthFiles/model_epoch_{epoch}")
    except Exception as err:
        print("Some exception occured: ",err)

    
    print("Evaluating Wait...")
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
            print("Error Converting to np and processing the x and y: ",err)
        # if i==0:
        #     print("Shape of each eval prediction: " ,predictions.shape)
        #     i =1
    final_val_loss = sum(val_loss)/len(val_loss)


    try:
        result = classification_report(x, y, target_names=label_names)
        # print(result)
        # Additional information to include with the report
        additional_info = f"/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/Saved_Models2/evaluations/eval_epoch{epoch}.txt"
        # Save the report with additional information to a text file
        with open(additional_info, 'w') as f:
            f.write(result)
    except:
        print("Error in evaluate metric compute: ",Exception)
    # print(x.shape,"--",x[:4])
    # print(y.shape,"--", y[:4])
    accuracy = accuracy_score(x,y)
    # Use accelerator.print to print only on the main process.
    try:
        # Log metrics to WandB for this epoch
        wandb.log({
            "validation_accuracy": accuracy,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
        })
    except Exception as err:
        print("Not able to log to wandb, ", err)

    accelerator.print(f"Epoch {epoch+1}/{num_epochs}: train_loss: {final_train_loss} val_loss: {final_val_loss} Val_Accuracy:{accuracy}")

print("\nTraining Done!")
print("$$"*100)
print("Work done mate")

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
#     print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
#     print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
#     print("FTP_PROXY:", os.environ.get("FTP_PROXY"))
# except Exception as err:
#     print("Error setting up the proxies: ", err)

# ####


# pathx = f"final_model_saved"
# trainer.save_model(pathx)
# print("Model saved at : ",pathx," cheers!!")

# print(f"Pushing to hub, you can find at {repo_url}")
# trainer.push_to_hub(repo_url)
# processor.push_to_hub(repo_url)

# print("Work done mate")
