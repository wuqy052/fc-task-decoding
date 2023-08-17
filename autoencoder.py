import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load and preprocess the FashionMNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()
# subset of data
x_train = x_train[0:500,:,:]
x_test = x_test[0:50,:,:]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the dimensions
input_dim = x_train.shape[1]
latent_dim = 32

# Build the autoencoder model
input_layer = Input(shape=(input_dim,))
encoder1 = Dense(256, activation='relu')(input_layer)
encoder2 = Dense(128, activation='relu')(encoder1)
encoder3 = Dense(latent_dim, activation='relu')(encoder2)

decoder1 = Dense(128, activation='relu')(encoder3)
decoder2 = Dense(256, activation='relu')(decoder1)
decoder3 = Dense(input_dim, activation='sigmoid')(decoder2)

autoencoder = Model(inputs = input_layer, outputs = decoder3)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Print the model info
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, x_train, # from x_train to x_train
                epochs=20, # number of times the entire dataset will be iterated through during training
                batch_size=100, # the number of samples that will be used in each update of the model's weights.
                shuffle=True, #  the training data will be shuffled before each epoch
                validation_data=(x_test, x_test))


decoded_imgs = autoencoder.predict(x_test)

# Define the file path to save the array
output_file = "decoded_images.npy"
# Save the array to a file
np.save(output_file, decoded_imgs)

output_file2 = "x_test.npy"
# Save the array to a file
np.save(output_file2, x_test)
