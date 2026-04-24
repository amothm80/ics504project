"""
convert wave files to STFT spectograms to images
"""
import os
import random
import numpy as np
import librosa
from PIL import Image

from tqdm import tqdm
from sklearn.model_selection import train_test_split

INPUT_DIR = "datasets"
OUTPUT_DIR = "dataset_stft_img"
SR = 16000
RANDOM_STATE = 42
IMG_SIZE = 300
DPI = 100

N_FFT = 512
HOP_LENGTH = 512


def get_files(root):
    """
    Load Files
    """
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

def split_dataset(files, labels):
    """
    Split Dataset
    """
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
    """
    add noise to sound files
    """
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, shift_max=0.2):
    """ time shift sound files"""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps=2):
    """ shift sound files shift """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate=1.2):
    """ time strech sound files """
    return librosa.effects.time_stretch(y, rate=rate)

def augment(y, sr, num_augments=2):
    """ augment sound files from one of 4 augmentation functions"""
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

def save_stft(y, sr, out_path):
    # S = librosa.feature.melspectrogram(y=y, sr=sr)
    # S_db = librosa.power_to_db(S, ref=np.max)

    MAX_LEN = SR * 3  # 3 seconds

    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    else:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    stft = librosa.stft(y, n_fft=N_FFT,
                            hop_length=HOP_LENGTH
                            )
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=1.0)
    stft_db = (stft_db + 80) / 80  # normalize to [0,1]

    img = Image.fromarray((stft_db * 255).astype(np.uint8))
    img = img.resize((300, 300))
    img.save(out_path)
    
def process_split(split, split_name, augment_data=False):
    """ process files into directories for each class """
    total = len(split)

    # for wav_path, label in split:
    for wav_path, label in tqdm(split, desc=f"{split_name.upper()}", total=total):
        out_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)

        y, sr = librosa.load(wav_path, sr=SR)

        samples = [y]
        if augment_data:
            num_aug = random.choice([1, 2])  # randomly 1 or 2 augmentations
            samples = augment(y, sr, num_augments=num_aug)
            # samples = augment(y, sr)

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
