## author @Yash Sharma, B20241
## imporying all the neccesary modules
## loading important libraries
from tqdm import tqdm
import torchaudio
import os
from datasets import  load_from_disk, concatenate_datasets
from transformers.file_utils import ModelOutput
import torch
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

# disable_caching()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## fetch trained model from huggingface hub
repo_url = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
model_name_or_path = repo_url
print("Device is: ",device)

cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash/cache"
final_path= os.path.join("/nlsasfs/home/nltm-st/sujitk/yash/datasets","saved_dataset.hf")
save_path = cache_dir
# i = 0
# final_path = f"/nlsasfs/home/nltm-st/sujitk/yash/datasets/disjointChunks/chunk_{i}"

## loading from saved dataset
dataset = load_from_disk(final_path)
print("DAtasets loaded succesfully!!")

# dataset = concatenate_datasets([dataset["train"],dataset["test"],dataset["validation"]])

## train test split
# temp = dataset.train_test_split(test_size=0.1)
dataset.cleanup_cache_files()
train_dataset = concatenate_datasets([dataset["train"],dataset["test"]])
eval_dataset = dataset["validation"]

# train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

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

# def preprocess_function(examples):
#     speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
#     target_list = [label_to_id(label, label_list) for label in examples[output_column]]

#     result = processor(speech_list, sampling_rate=target_sampling_rate)
#     result["labels"] = list(target_list)

#     return result
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
    batch_size=256,
    batched=True,
    num_proc=48,
    keep_in_memory=True
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=256,
    batched=True,
    num_proc=48,
    keep_in_memory=True
)
    
label2id={label: i for i, label in enumerate(label_list)}
id2label={i: label for i, label in enumerate(label_list)}

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
checkpt = os.path.join(save_path, "chkpt-wav2vec2-large-xlsr-indian-language-classification-featureExtractor")
proxy_url = "http://proxy-10g.10g.siddhi.param:9090"



early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement to wait before stopping
    early_stopping_threshold=0.0,  # Minimum improvement required to consider it as an improvement
)

model.to(device)

training_args = TrainingArguments(
    output_dir=model_out_dir,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    overwrite_output_dir=True,
    num_train_epochs=250,
    save_steps=1*eval_steps,  # Save checkpoints every 1000 steps
    logging_steps=200,
    learning_rate=5e-5,
    save_total_limit=5,
    fp16=True,
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
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)


# Initialize Wandb with your API key
wandb.login(key="690a936311e63ff7c923d0a2992105f537cd7c59")

train_result = trainer.train()
print("\nTraining Done!")
print("$$"*100)

## finding the metrics
# After the training/evaluation, you can use the following code to save metrics.
# Compute train results
metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

# Save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Compute evaluation results
metrics = trainer.evaluate()
max_val_samples = len(eval_dataset)
metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

# Save evaluation results
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# Save logs to the output directory
trainer.save_state()

## logging into the huggingface to push to the hub
secret_value_0 = "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"
login(secret_value_0)



pathx = f"final_model_saved"
trainer.save_model(pathx)
print("Model saved at : ",pathx," cheers!!")

print(f"Pushing to hub, you can find at {repo_url}")
trainer.push_to_hub(repo_url)
processor.push_to_hub(repo_url)

print("Work done mate")
