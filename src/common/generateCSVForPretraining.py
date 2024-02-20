import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Directories containing the WAV files
target = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/pretraining-dataset"
directory = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/pretrain-dataset-unsup"

# Initialize a dictionary to store the data
data_dict = {'PATH': []}

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(directory, filename)
        data_dict['PATH'].append(filepath)

# Convert dictionary to DataFrame
df = pd.DataFrame(data_dict)

# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Write train data to CSV file
train_csv_file = os.path.join(target,'train.csv')
train_df.to_csv(train_csv_file, index=False)
print("Train CSV file generated successfully:", train_csv_file)

# Write validation data to CSV file
val_csv_file = os.path.join(target,'val.csv')
val_df.to_csv(val_csv_file, index=False)
print("Validation CSV file generated successfully:", val_csv_file)

print(train_df.head())
print(val_df.head())
print("Done saving.")
