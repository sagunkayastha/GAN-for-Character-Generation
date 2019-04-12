
from __future__ import print_function, division
import argparse
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np

class DCGAN():
    def __init__(self,to_pre,train_folder):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.to_pre = to_pre
        self.train_folder = train_folder
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8*8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _) = self.load_data()
        print(f'Total Training Images = {len(X_train)}')
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        try:
            os.mkdir('predict/'+self.to_pre)
            print(f'{self.to_pre} folder created')
        except:
            print(self.to_pre+' directory already exist')

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------


            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                try:
                    os.mkdir('weights/'+self.to_pre)
                except:
                    pass

                self.generator.save_weights(f'weights/{self.to_pre}/{epoch}_gen.h5')
                self.discriminator.save_weights(f'weights/{self.to_pre}/{epoch}_dis.h5')
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 1, 1
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        destRGB = cv2.cvtColor(gen_imgs[0], cv2.COLOR_BGR2RGB)
        try:
            to_save_dir = 'predict/'+self.to_pre
        except:
            to_save_dir = 'predict/all'

        plt.imsave(to_save_dir+"/%d.png" % epoch, destRGB )




    def load_data(self):

        print(self.train_folder+self.to_pre+'/')
        images = os.listdir(self.train_folder+self.to_pre+'/')
        num_images = len(images)
        train_x = np.zeros((num_images,32,32,3 ),dtype='uint8')
        print(self.train_folder+self.to_pre+'/')
        for i,name in enumerate(images):
            # print(name)
            im = cv2.imread('dataset2/train/'+self.to_pre+'/'+name)

            train_x[i]  = im

        return(train_x, [])

    def predict_(self,epochs,number_of_images):
        for epoch in epochs:

            if 'Final_Predict' not in os.listdir("predict"):
                os.mkdir(f'predict/Final_Predict/')

            if self.to_pre not in os.listdir("predict/Final_Predict/"):
                os.mkdir(f'predict/Final_Predict/{self.to_pre}')

            try:
                os.mkdir(f'predict/Final_Predict/{self.to_pre}/{epoch}')
            except:
                pass
            to_save_dir = f'predict/Final_Predict/{self.to_pre}/{epoch}'

            for i in range(number_of_images):
                noise = np.random.normal(0, 1, (1 * 1, self.latent_dim))



                try:
                    weights_folder ='weights/'+self.to_pre
                    self.generator.load_weights(f'weights/{self.to_pre}/{epoch}_gen.h5')
                    self.discriminator.load_weights(f'weights/{self.to_pre}/{epoch}_dis.h5')

                except:
                    print('Weights file not found')

                gen_imgs = self.generator.predict(noise)
                gen_imgs = 0.5 * gen_imgs + 0.5

                """POST PROCESS"""
                to_save_dir = f'predict/Final_Predict/{self.to_pre}/{epoch}'
                destRGB = cv2.cvtColor(gen_imgs[0], cv2.COLOR_BGR2RGB)
                plt.imsave(to_save_dir+"/%i.png" % i, destRGB )





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str,
                        help='TRAIN', required = True)
    # parser.add_argument('-P', type=str,
    #                     help='PREDICT')
    parser.add_argument('-chr', type=str,
                        help='Character to train/predict')
    parser.add_argument('-train_folder', type=str,
                        help='Train Folder')
    parser.add_argument('-epochs', type=int, default=5000,
                        help='number of epochs to train')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('-save_interval', type=int, default=50,
                        help='save_interval_for_weights')
    parser.add_argument('-n2p', type=int, default=100,
                        help='number of images to predic')
    parser.add_argument('-pepochs', type=str, default="3900",
                    help='epoch numnbers to predict')
                        #]
    args = parser.parse_args()


    to_pre = args.chr
    pepochs = args.pepochs.split(',')
    pepochs = [int(i) for i in pepochs]

    if args.mode == 'TRAIN':
        dcgan = DCGAN(to_pre,train_folder = args.train_folder)
        dcgan.train(epochs=args.epochs, batch_size=args.batch_size, save_interval=args.save_interval, )

    elif args.mode == 'PREDICT':
        dcgan = DCGAN(to_pre)
        dcgan.predict_(pepochs,args.n2p)


if __name__ == '__main__':
    if 'predict' not in os.listdir():
        os.mkdir('predict/')
    if 'weights' not in os.listdir():
        os.mkdir('weights/')


    main()
