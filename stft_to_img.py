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
OUTPUT_DIR = "dataset_stft_img"

SR = 16000
N_FFT = 2048
HOP_LENGTH = 512

IMG_SIZE = 300   # pixels
DPI = 100        # 3 inches * 100 dpi = 300 px

# =========================
# FUNCTION
# =========================
def wav_to_stft(input_path, output_path):
    try:
        y, sr = librosa.load(input_path, sr=SR)

        stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        fig = plt.figure(figsize=(IMG_SIZE / DPI, IMG_SIZE / DPI), dpi=DPI)
        ax = plt.axes([0, 0, 1, 1])  # remove margins completely

        librosa.display.specshow(
            stft_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis=None,
            y_axis=None
        )

        ax.set_axis_off()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=DPI)
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


# =========================
# MAIN
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
        output_path = os.path.splitext(output_path)[0] + ".png"

        wav_to_stft(wav_path, output_path)


if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)