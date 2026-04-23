import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "datasets"
OUTPUT_DIR = "dataset_mfcc_img"

SR = 16000
N_MFCC = 128
N_FFT = 512
HOP_LENGTH = 512

IMG_SIZE = 300
DPI = 100

# =========================
# AUGMENTATIONS
# =========================
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate=1.2):
    return librosa.effects.time_stretch(y, rate=rate)

# =========================
# FEATURE → IMAGE
# =========================
def save_mfcc(y, sr, output_path):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Remove energy coefficient
    mfcc = mfcc[1:, :]



    fig = plt.figure(figsize=(IMG_SIZE / DPI, IMG_SIZE / DPI), dpi=DPI)
    ax = plt.axes([0, 0, 1, 1])

    librosa.display.specshow(
        mfcc,
        sr=sr,
        # hop_length=HOP_LENGTH,
        cmap='magma',
        vmin=-3,
        vmax=3
    )

    plt.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# =========================
# PROCESS ONE FILE
# =========================
def process_file(input_path, output_base_path):
    try:
        y, sr = librosa.load(input_path, sr=SR)

        # Define augmentations
        augmented = {
            "original": y,
            "shift": time_shift(y),
            "pitch": pitch_shift(y, sr),
            "noise": add_noise(y),
            "stretch": time_stretch(y)
        }

        # Save each version
        for aug_name, y_aug in augmented.items():
            out_path = f"{output_base_path}_{aug_name}.png"
            save_mfcc(y_aug, sr, out_path)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# =========================
# MAIN LOOP
# =========================
def process_dataset(input_dir, output_dir):
    wav_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    print(f"Found {len(wav_files)} wav files.")

    for wav_path in tqdm(wav_files):
        relative_path = os.path.relpath(wav_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        # remove extension
        output_base = os.path.splitext(output_path)[0]

        process_file(wav_path, output_base)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)