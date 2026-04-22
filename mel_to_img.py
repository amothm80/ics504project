import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "datasets"      # root folder of wav files
OUTPUT_DIR = "datasets_img"   # root folder for mel spectrograms

SR = 22050          # sampling rate
N_MELS = 128        # number of mel bands
N_FFT = 2048
HOP_LENGTH = 512

# =========================
# FUNCTION: process one file
# =========================
def wav_to_mel(input_path, output_path):
    try:
        y, sr = librosa.load(input_path, sr=SR)

        # Generate mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Save as image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


# =========================
# MAIN LOOP
# =========================
def process_dataset(input_dir, output_dir):
    wav_files = []

    # Collect all wav files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    print(f"Found {len(wav_files)} wav files.")

    for wav_path in tqdm(wav_files):
        # Create mirrored output path
        relative_path = os.path.relpath(wav_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        # Change extension to .png
        output_path = os.path.splitext(output_path)[0] + ".png"

        wav_to_mel(wav_path, output_path)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)