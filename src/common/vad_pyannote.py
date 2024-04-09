import argparse
import os
import pickle
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import matplotlib.pyplot as plt
import numpy as np
import sys

VAD_SUBDIR='dev/'
SEG_SUBDIR='seg/'

def pyannote_vad(input_wav, output_dir):
    # output_dir = '/home1/pratik/lang_diarization/output/dev/'
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "onset": 0.5, "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(input_wav)
    nfile = os.path.splitext(os.path.basename(input_wav))[0]
    tdir = os.path.join(output_dir,VAD_SUBDIR) ; 
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir , f"pyannote_vad_{nfile}.pkl"), 'wb') as f:
        pickle.dump(vad, f)

def load_pyannote(f_name):
    segments_dict = np.load(f_name, allow_pickle=True)
    d = [i for i in segments_dict.get_timeline()]
    d = [[j for j in i] for i in d]
    return d

def results_segments(out_dir):
    vad_dir_path = os.path.join(out_dir,VAD_SUBDIR );
    file_dir = os.listdir(vad_dir_path)
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(out_dir , SEG_SUBDIR) # '../output/dev_final_seg/'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(file_dir)):
        if file_dir[i].split("_")[0] == 'pyannote':
            segment = load_pyannote(vad_dir_path + "/" + file_dir[i])
            string_content = "".join(str(k[0]) + " " + str(k[1]) + "\n" for k in segment)
            f_name = os.path.join(output_dir, file_dir[i][-8:-4] + ".pyannote.segment")
            
            with open(f_name, 'w') as d:
                d.write(string_content)

def process_folder(in_dir,out_dir):
    #folder_path = '/home1/pratik/lang_diarization/data'
    for file_name in os.listdir(in_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(in_dir, file_name)
            pyannote_vad(file_path,out_dir)

def main():
    global VAD_SUBDIR;
    global SEG_SUBDIR;
    print(os.path.isdir(sys.argv[1]))
    if(os.path.isdir(sys.argv[1])):
        print('Processing folder  = ',sys.argv[1]);
        print('Output written to  = ',sys.argv[2]);
        INPUT_FOLDER=sys.argv[1]; # /home1/pratik/lang_diarization/data/dev
        OUTPUT_FOLDER=sys.argv[2];# /home1/pratik/lang_diarization/output 
        VAD_SUBDIR=sys.argv[3]  # dev 
        SEG_SUBDIR=sys.argv[4]  # devseg 
        print('sub directories are ', VAD_SUBDIR, SEG_SUBDIR ); 
    else:
        print('Cannot find folder = ',sys.argv[1]);
        exit(0);
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    results_segments(OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
