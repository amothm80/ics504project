import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only errors are logged
os.environ['TF_GPU_ALLOCATOR'] ='cuda_malloc_async'

import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import ops

# TF imports related to tf.data preprocessing
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow.keras.utils import plot_model

keras.utils.set_random_seed(42)
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 100
SAMPLE_RATE = 16000
OUT_SEQ_LEN = 72000

keras.backend.clear_session(free_memory=True)
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(directory='dataset_stft_img',
image_size=(300, 300),
subset='both',
batch_size=BATCH_SIZE,
validation_split=0.2,
seed=42
)

print(train_ds)