from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import glob
import random

n_colors = 3

def generator_model():
    model = Sequential()

    model.add(Dense(1024, input_shape=(100,)))
    model.add(Activation('tanh'))

    model.add(Dense(128 * 16 * 16))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((16, 16, 128)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(n_colors, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=(64, 64, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def image_batch(batch_size):
    files = glob.glob("./in_images/**/*.png", recursive=True)
    files = random.sample(files, batch_size)
    # print(files)
    res = []
    for path in files:
        img = Image.open(path)
        img = img.resize((64, 64))
        arr = np.array(img)
        arr = (arr - 127.5) / 127.5
        arr.resize((64, 64, n_colors))
        res.append(arr)
    return np.array(res)

def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, n_colors))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def main():
    batch_size = 55
    discriminator = discriminator_model()
    generator = generator_model()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    set_trainable(discriminator, False)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    print(generator.summary())
    print(discriminator_on_generator.summary())

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    print(discriminator.summary())

    for i in range(30 * 1000):
        batch_images = image_batch(batch_size)
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X, y)
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
        if i % 100 == 0:
            print("step %d d_loss, g_loss : %g %g" % (i, d_loss, g_loss))
            image = combine_images(generated_images)
            os.system('mkdir -p ./gen_images')
            image.save("./gen_images/gen%05d.png" % i)
            # generator.save_weights('generator.h5', True)
            # discriminator.save_weights('discriminator.h5', True)

main()
