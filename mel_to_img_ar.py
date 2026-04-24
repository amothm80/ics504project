import os
import random
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
INPUT_DIR = "datasets_ar"          # original wav dataset
OUTPUT_DIR = "dataset_mel_img_ar" # output folder
# IMG_SIZE = (3, 3)             # inches (controls resolution)
SR = 16000
RANDOM_STATE = 42
IMG_SIZE = 300
DPI = 100

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
def augment(y, sr, num_augments=2):
    augmented = [y]  # always include original

    # list of augmentation functions
    aug_funcs = [
        lambda x: time_stretch(x, rate=random.uniform(0.85, 1.15)),
        lambda x: pitch_shift(x, sr, n_steps=random.uniform(-2, 2)),
        lambda x: time_shift(x, shift_max=0.2),
        lambda x: add_noise(x, noise_factor=random.uniform(0.002, 0.01)),
    ]

    # randomly choose augmentations
    selected = random.sample(aug_funcs, k=num_augments)

    for func in selected:
        try:
            augmented.append(func(y))
        except:
            continue  # skip failed augmentations safely

    return augmented

# =========================
# STEP 4: SAVE MEL SPECTROGRAM
# =========================
def save_mel(y, sr, out_path):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -80, 0)
    S_db = (S_db + 80) / 80

    img = Image.fromarray((S_db * 255).astype(np.uint8))
    img = img.resize((300, 300))
    img.save(out_path)
    # plt.figure(figsize=(IMG_SIZE / DPI, IMG_SIZE / DPI), dpi=DPI)

    # # plt.figure(figsize=IMG_SIZE)
    # plt.axis("off")
    # librosa.display.specshow(S_db, sr=sr)
    # plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    # plt.close()

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

        MAX_LEN = SR * 3  # 3 seconds

        if len(y) > MAX_LEN:
            y = y[:MAX_LEN]
        else:
            y = np.pad(y, (0, MAX_LEN - len(y)))

        samples = [y]
        if augment_data:
            num_aug = random.choice([1, 2])  # randomly 1 or 2 augmentations
            samples = augment(y, sr, num_augments=num_aug)

        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        for i, sample in enumerate(samples):
            out_path = os.path.join(out_dir, f"{base_name}_{i}.png")
            save_mel(sample, sr, out_path)

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