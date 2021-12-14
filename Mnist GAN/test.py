from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

model_input = sys.argv[1]
save_path = sys.argv[2]

np.random.seed(10)
noise_dim = 100
batch_size = 10
steps_per_epoch = 3750
epochs = 10

if not os.path.isdir(save_path):
    os.mkdir(save_path)

img_rows, img_cols, channels = 28, 28, 1

generator = load_model(model_input)

def save_images(noise):
    
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        if channels == 1:
            plt.imshow(image.reshape((img_rows, img_cols)), cmap='gray')
        else:
            plt.imshow(image.reshape((img_rows, img_cols, channels)))
        plt.axis('off')
    
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/gan-images.png')

noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
#print(noise)

save_images(noise)