#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

import sys
sys.path.append("/app/src/common")
from Model import *
import os
from SilenceRemover import *
from datasets import Dataset
from multiprocess import set_start_method
import numpy as np
import torch
from transformers import  AutoConfig, Wav2Vec2Processor
import librosa
from torch import mps
import torch.nn.functional as F
from sklearn.metrics import classification_report
import pandas as pd
import IPython.display as ipd
import matplotlib.pyplot as plt
import torchaudio

## Selecting the device
def whatDevice():
    if  torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"
device = whatDevice()
print(f"Device: {device}")

############################################################################################################################################
## Intializations
############################################################################################################################################
directory = "/Users/yash/Desktop/MTP-2k23-24/Bhashini_Test_Data"
### Intializing models
## for wave2vec2
model_name_or_path = "yashcode00/wav2vec2-large-xlsr-indian-language-classification-featureExtractor"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model_wave2vec2 = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
target_sampling_rate = processor.feature_extractor.sampling_rate
processor.feature_extractor.return_attention_mask = True
label_list  = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'tam', 'tel']
lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'tam': 9, 'tel': 10,'pun': 10}
id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'tam', 10: 'tel'}
input_column = 'path'
output_column = 'true_label'
default_path = "dummy-save-folder"
window_size = 48000  #### window size of 3 seconds
hop_length_seconds = 0.5
# Calculate the hop size in samples
hop_size = int(hop_length_seconds * target_sampling_rate)  # Adjust 'sample_rate' as needed

############################################################################################################################################
############################################################################################################################################

class ModelForApp:
    def __init__(self) -> None:
        pass
    def speech_file_to_array_fn(self, path: str):
        audio, sr =  librosa.load(path, sr = target_sampling_rate, mono = True)
        clips = librosa.effects.split(audio, frame_length=8000, top_db=10) ## setting frame length 8000 and topdb=10 gave desirable results
        wav_data = []
        for c in clips:
            data = audio[c[0]: c[1]]
            wav_data.extend(data)
        return wav_data

    ## function to store the hidden feature representation from the last layer of wave2vec2
    def predictFrames(self, frames):
        features = processor(frames, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)
        # print(f"shape of the processed input is: {input_values.shape}")
        try:
            with torch.no_grad():
                logits = model_wave2vec2(input_values, attention_mask=attention_mask).logits 
        except Exception as err:
            print(f"Error -> {err} \nSKIPPED! Input Length was: {len(frames[-1])} and features len was : {input_values.shape}")
        return logits

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predictOneAudio(self, path):
        x = self.speech_file_to_array_fn(path)
        # Generate overlapping frames
        frames = [np.asarray(x[i:i+window_size]).astype('float32') for i in range(0, len(x) - window_size + 1, hop_size)]
        # print(f"Length of frames: {len(frames)}")
        # print(frames)
        if len(frames) == 0:
            print("Error! Invalid Audio file, please check the audio again..")
            return None
        if len(frames[-1]) < 100:
            print(f"Last element has small length of {len(frames[-1])} while it shall be {len(frames[0])}, Dropping!")
            frames.pop()
        logits = self.predictFrames(frames)
        # Combine logits across frames
        logits = np.vstack(logits)  # Stack logits vertically to combine predictions across frames
        # Take mode to get the final prediction
        final_prediction = np.argmax(np.bincount(np.argmax(logits, axis=-1)))
        # Calculate probabilities using softmax
        prob_distribution = self.softmax(logits)
        final_probs = np.mean(prob_distribution, axis=0)
        # Format outputs
        outputs = {'confidences':{}}
        outputs['predicted_class'] = id2lang[final_prediction]
        for i, prob in enumerate(final_probs):
            outputs['confidences'][id2lang[i]] = f"{prob * 100:.1f}%"
        return outputs

    def predict(self, path):
        orig_audio, sr = torchaudio.load(path)
        print(f"Original sampling rate is {sr}")
        speech = self.speech_file_to_array_fn(path)    
        # ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=target_sampling_rate))
        out = self.predictOneAudio(path)
        out["original_sr"] = sr 
        out["original_audio"] = orig_audio.numpy()[0].tolist()
        out["silence_removed_audio"] = np.array(speech).tolist()
        # print(f"{type(orig_audio.numpy())} -> {orig_audio.numpy()}")
        # print(f"{type(np.array(speech)) } -> {np.array(speech)}")
        return out