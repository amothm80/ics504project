# Project Setup

## Audio Preprocessing

1. Install SoX on Linux:
   - `sudo apt-get install sox`

2. Separate the dataset into labeled folders using your preprocessing script.

3. Standardize all audio files in the dataset:
   - `sox file.wav -r 16000 -c 1 -b 16 fixed_file.wav`

## Python Requirements

Install the Python dependencies from the requirements file:

- `pip install -r requirments.txt`
