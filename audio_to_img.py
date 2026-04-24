import os
import random
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# =========================
# CONFIG
# =========================
SR = 16000
RANDOM_STATE = 42
IMG_SIZE = 300
DPI = 100

# MFCC-specific
N_MFCC = 40
N_FFT_MFCC = 512
HOP_LENGTH_MFCC = 512

# STFT-specific
N_FFT_STFT = 512
HOP_LENGTH_STFT = 512

# Mel-specific
N_FFT_MEL = 1024
HOP_LENGTH_MEL = 512
N_MELS = 128

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

def augment(y, sr, num_augments=2):
    augmented = [y]  # always include original
    aug_funcs = [
        lambda x: time_stretch(x, rate=random.uniform(0.85, 1.15)),
        lambda x: pitch_shift(x, sr, n_steps=random.uniform(-2, 2)),
        lambda x: time_shift(x, shift_max=0.2),
        lambda x: add_noise(x, noise_factor=random.uniform(0.002, 0.01)),
    ]
    selected = random.sample(aug_funcs, k=num_augments)
    for func in selected:
        try:
            augmented.append(func(y))
        except:
            continue
    return augmented

# =========================
# STEP 3: PAD / TRIM
# =========================
def fix_length(y, sr, duration=3):
    max_len = sr * duration
    if len(y) > max_len:
        return y[:max_len]
    return np.pad(y, (0, max_len - len(y)))

# =========================
# STEP 4: FEATURE → IMAGE
# =========================
def save_mel(y, sr, out_path):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT_MEL, hop_length=HOP_LENGTH_MEL, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -80, 0)
    S_db = (S_db + 80) / 80
    img = Image.fromarray((S_db * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(out_path)

def save_mfcc(y, sr, out_path):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT_MFCC, hop_length=HOP_LENGTH_MFCC
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    mfcc = np.clip(mfcc, -3, 3)
    mfcc = (mfcc + 3) / 6
    img = Image.fromarray((mfcc * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(out_path)

def save_stft(y, sr, out_path):
    stft = librosa.stft(y, n_fft=N_FFT_STFT, hop_length=HOP_LENGTH_STFT)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=1.0)
    stft_db = (stft_db + 80) / 80
    img = Image.fromarray((stft_db * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(out_path)

SAVE_FN = {
    "mel":  save_mel,
    "mfcc": save_mfcc,
    "stft": save_stft,
}

# =========================
# STEP 5: PROCESS SPLIT
# =========================
def process_split(split, split_name, output_dir, feature, augment_data=False):
    save_fn = SAVE_FN[feature]
    for wav_path, label in tqdm(split, desc=f"{split_name.upper()}", total=len(split)):
        out_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(out_dir, exist_ok=True)

        y, sr = librosa.load(wav_path, sr=SR)
        y = fix_length(y, sr)

        samples = [y]
        if augment_data:
            num_aug = random.choice([1, 2])
            samples = augment(y, sr, num_augments=num_aug)

        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        for i, sample in enumerate(samples):
            out_path = os.path.join(out_dir, f"{base_name}_{i}.png")
            save_fn(sample, sr, out_path)

# =========================
# MAIN PIPELINE
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV dataset to spectrogram images."
    )
    parser.add_argument(
        "--feature",
        choices=["mel", "mfcc", "stft"],
        required=True,
        help="Feature type to extract: mel | mfcc | stft",
    )
    parser.add_argument(
        "--arabic",
        action="store_true",
        help="Use Arabic dataset (datasets_ar → dataset_<feature>_img_ar)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override input directory (default: datasets or datasets_ar)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: dataset_<feature>_img[_ar])",
    )
    args = parser.parse_args()

    suffix = "_ar" if args.arabic else ""
    input_dir  = args.input_dir  or f"datasets{suffix}"
    output_dir = args.output_dir or f"dataset_{args.feature}_img{suffix}"

    print(f"Feature  : {args.feature.upper()}")
    print(f"Arabic   : {args.arabic}")
    print(f"Input    : {input_dir}")
    print(f"Output   : {output_dir}")

    print("\nLoading dataset...")
    files, labels = get_files(input_dir)
    print(f"Total files: {len(files)}")

    print("Splitting dataset...")
    train, val, test = split_dataset(files, labels)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("Processing TRAIN (with augmentation)...")
    process_split(train, "train", output_dir, args.feature, augment_data=True)

    print("Processing VAL (no augmentation)...")
    process_split(val, "val", output_dir, args.feature, augment_data=False)

    print("Processing TEST (no augmentation)...")
    process_split(test, "test", output_dir, args.feature, augment_data=False)

    print("Done.")

if __name__ == "__main__":
    main()