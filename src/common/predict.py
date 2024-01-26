## loading important libraries
from tqdm import tqdm
import torchaudio
import os
import IPython.display as ipd
from SilenceRemover import *

class PredictOne:
    def __init__(self, audioPath: str = None) -> None:
        self.audioPath = audioPath
    
    def SilenceAudio(self):
        try:
            ## try to load the file if not broken
            speech, sr = torchaudio.load(self.audioPath)
            silencedAudio = RemoveSilence(finalPath, AudioPath)
            count = count +1;
        except Exception as e:
            print(str(path), e)
            pass
