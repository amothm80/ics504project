import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_DIR = "datasets"          # original wav dataset
OUTPUT_DIR = "dataset_stft_img" # output folder
# IMG_SIZE = (3, 3)             # inches (controls resolution)
SR = 16000
RANDOM_STATE = 42
IMG_SIZE = 300
DPI = 100

N_FFT = 512
HOP_LENGTH = 512



# =========================
# STEP 1: LOAD FILES
# =========================
def get_files(root):
    files = []
    labels = []
    for label in os.listdir(root):
        class_dir = os.path.join(root, label)
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.endswith(".wav"):
                files.append((os.path.join(class_dir, f), label))
                labels.append(label)
    return files, labels

# =========================
# STEP 2: SPLIT BEFORE AUGMENTATION
# =========================
def split_dataset(files, labels):
    train, temp = train_test_split(
        files, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
    )

    temp_labels = [label for _, label in temp]

    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp_labels, random_state=RANDOM_STATE
    )

    return train, val, test

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
# STEP 3: AUGMENTATION (TRAIN ONLY)
# =========================
def augment(y, sr):
    augmented = []

    # original
    augmented.append(y)

    #time stretch
    augmented.append(time_stretch(y, rate=0.9))
    augmented.append(time_stretch(y, rate=1.1))

    #pitch shift
    augmented.append(pitch_shift(y, sr))

    #time shift
    augmented.append(time_shift(y))

    # noise
    # augmented.append(add_noise(y))

    return augmented

# =========================
# STEP 4: SAVE MEL SPECTROGRAM
# =========================
def save_stft(y, sr, out_path):
    # S = librosa.feature.melspectrogram(y=y, sr=sr)
    # S_db = librosa.power_to_db(S, ref=np.max)

    stft = librosa.stft(y, n_fft=N_FFT,
                            hop_length=HOP_LENGTH
                            )
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(IMG_SIZE / DPI, IMG_SIZE / DPI), dpi=DPI)

    # plt.figure(figsize=IMG_SIZE)
    plt.axis("off")
    librosa.display.specshow(stft_db, sr=sr)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# =========================
# STEP 5: PROCESS SPLIT
# =========================
def process_split(split, split_name, augment_data=False):
    total = len(split)

    # for wav_path, label in split:
    for wav_path, label in tqdm(split, desc=f"{split_name.upper()}", total=total):
        out_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)

        y, sr = librosa.load(wav_path, sr=SR)

        samples = [y]
        if augment_data:
            samples = augment(y, sr)

        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        for i, sample in enumerate(samples):
            out_path = os.path.join(out_dir, f"{base_name}_{i}.png")
            save_stft(sample, sr, out_path)

# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading dataset...")
    files, labels = get_files(INPUT_DIR)

    print(f"Total files: {len(files)}")

    print("Splitting dataset...")
    train, val, test = split_dataset(files, labels)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("Processing TRAIN (with augmentation)...")
    process_split(train, "train", augment_data=True)

    print("Processing VAL (no augmentation)...")
    process_split(val, "val", augment_data=False)

    print("Processing TEST (no augmentation)...")
    process_split(test, "test", augment_data=False)

    print("Done.")

if __name__ == "__main__":
    main()