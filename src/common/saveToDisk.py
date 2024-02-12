## author @Yash Sharma, B20241
## imporying all the neccesary modules
## loading important libraries
import pandas as pd
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
from datasets import load_dataset
# is_apex_available

cache_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/csv"
final_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2"


## loading data from disk
print("Loading the data from the disk.. wait")
# # loading the autdio dataset from the directory
directory  = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/combined-resampled_data_SilenceAndTwoSecond"
print(f"Current directory {directory}")
print(os.listdir(directory))
data = []

for path in tqdm(os.listdir(directory)):
    # now eplore the inner folder ,
    #  path is actually the audio language
    if path in ['eng', 'tel' ,'mar' ,'odi' ,'asm' ,'guj' ,'hin' ,'tam', 'kan' ,'mal', 'ben','pun']:
        print(f"Processing {path}")
        pathHere = os.path.join(directory, path);
        count = 0
        ## Now expploring all the available audio files inside 
        ## and if not corrupted storing then in dataframe 
        for audioSamples in os.listdir(pathHere):
            ## extracto all req info
            name = audioSamples.split(".")[0]
            finalPath = os.path.join(pathHere, audioSamples);
            try:
                # There are some broken files
                s, sr = torchaudio.load(finalPath)
                ## dummy path
                data.append({
                    "name": name,
                    "path": finalPath,
                    "sampling_rate": sr,
                    "language": path,
                });
                count = count +1;
            except Exception as e:
                print(str(path), e)
                pass
        print(f'Total {count} samples loaded of {path} language.')

## now lets form a dataframe from the data array
df = pd.DataFrame(data)
print("Total length of the Dataset: ", len(data))
print(df.head())
print(df.iloc[0])

## ecpplore dataset stats
print("Labels: ", df["language"].unique())
print()
df.groupby("language").count()[["path"]]

# cache_dir = "/kaggle/working"

# Split the data into train, eval, and test sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["language"])

# Reset the index for all dataframes
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
# test_df = test_df.reset_index(drop=True)

# Save the train, eval, and test data as CSV files
train_df.to_csv(f"{cache_dir}/train.csv", sep="\t", encoding="utf-8", index=False)
eval_df.to_csv(f"{cache_dir}/eval.csv", sep="\t", encoding="utf-8", index=False)
# test_df.to_csv(f"{cache_dir}/test.csv", sep="\t", encoding="utf-8", index=False)

print("Train df is ", train_df.shape)
print("Validation df is ",eval_df.shape)
# print("Test df is ",test_df.shape)


## ############### loading the data ######################
# Loading the created dataset using datasets
# !pip install -q datasets==2.14.4

data_files = {
    "train": f"{cache_dir}/train.csv", 
    "validation": f"{cache_dir}/eval.csv",
    # "test": f"{cache_dir}/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
# test_dataset = dataset["test"]

print(train_dataset)
print(eval_dataset)
# print(test_dataset)

final_path= os.path.join(final_path,"combined-saved-dataset.hf")
print("Saving the dataset to be further use at ",final_path)
dataset.save_to_disk(final_path)

################# Done saving ##############################
print("Work done mate")
