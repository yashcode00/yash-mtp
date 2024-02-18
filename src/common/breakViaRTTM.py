import os
from pydub import AudioSegment

# Function to parse RTTM file
def parse_rttm(rttm_file):
    segments = []
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segments.append((float(parts[3]), float(parts[4]), parts[7]))
    return segments

# Function to split audio based on segments
def split_audio(audio_file, segments, output_dir):
    audio = AudioSegment.from_wav(audio_file)
    for start, duration, lang in segments:
        lang_dir = os.path.join(output_dir, "eng" if lang=="L1" else "not-eng")
        os.makedirs(lang_dir, exist_ok=True)
        segment_audio = audio[int(start * 1000):int((start + duration) * 1000)]
        segment_audio.export(os.path.join(lang_dir, f"{os.path.basename(audio_file).split('.')[0]}_{start}_{duration}.wav"), format="wav")

# Main function
def main():
    audio_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
    rttm_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/Displace2024_dev_labels_supervised/Labels/Track2_LD"
    output_dir = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/displace-challenge/displace-audioFromRTTM"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".wav"):
            base_name = audio_file.split(".")[0]
            rttm_file = os.path.join(rttm_dir,base_name + "_LANGUAGE.rttm")
            if os.path.isfile(rttm_file):
                segments = parse_rttm(rttm_file)
                audio_path = os.path.join(audio_dir, audio_file)
                split_audio(audio_path, segments, output_dir)
            else:
                print(f"Unable to find rttm {rttm_file}")

if __name__ == "__main__":
    main()
