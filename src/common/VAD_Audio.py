import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

target_sr = 16000

# Function to perform energy-based voice activity detection and extract voiced segments
def extract_voiced_segments(audio, sample_rate, frame_duration=20):
    frame_size = int(sample_rate * frame_duration / 1000)  # Frame size in samples
    hop_size = int(frame_size)  # Hop size in samples

    # Calculate energy for each frame
    energy = np.array([sum(frame ** 2) for frame in frames(audio, frame_size, hop_size)])

    # Calculate frame-level energy using squared values
    # frame_energy = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_size)**2

    # Apply VAD by thresholding the energy
    energy_threshold = np.percentile(energy, 10)  # Adjust the percentile to set the threshold
    
    # Find segments with energy above the threshold
    voiced_indices = np.where(energy > energy_threshold)[0]
    voiced_region = np.array(energy > energy_threshold)
    voiced_segments = np.concatenate([audio[idx * hop_size:(idx * hop_size + frame_size)] for idx in voiced_indices])

    return voiced_segments, voiced_region

# Helper function to split audio into frames
def frames(audio, frame_size, hop_size):
    num_frames = (len(audio) - frame_size) // hop_size + 1
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        yield audio[start:end]

def test(wav_file):
    ### To test the code
    frame_size = 30  # Adjust the frame length as desired
    audio, sample_rate = librosa.load(wav_file)
    voiced_audio, voiced_region = extract_voiced_segments(audio, sample_rate, frame_duration=frame_size)

    # Resample to the target sample rate and convert to floating-point
    silencedAudio = librosa.resample(voiced_audio.astype(float), orig_sr=sample_rate, target_sr=target_sr)


    ### To save the voiced audio in wav file
    # output_filename = "vad_sa2.wav"
    # sf.write(output_filename, voiced_audio, sample_rate)

    # Display/Plot the original signal and VAD signal
    plt.figure(figsize=(10, 4))
    plt.subplot(3, 1, 1)
    plt.plot(audio)
    plt.title('Original Signal')
    plt.subplot(3, 1, 2)
    plt.plot(voiced_region)
    plt.title('VAD Region')
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    plt.plot(voiced_audio)
    plt.title('VAD Signal')
    plt.show()

    return voiced_audio

    