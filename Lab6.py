# example of loading the cifar10 dataset
# Expected shapes for training dataset and test dataset
# Train (50000, 32, 32, 3) (50000, 1)
# Test (10000, 32, 32, 3) (10000, 1)

# preprocess and load the data 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.startswith("resized") and (filename.endswith(".jpg") or filename.endswith(".png")):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            #image = image.resize((64, 64))  # Resize the image to desired dimensions - no need to resize since we already have separate script that did that
            image = img_to_array(image)
            images.append(image)
    return np.array(images)

# Specify the directory containing the images
image_directory = "/Python/470/eclipse_gan/frames"

# Load images from the directory
dataset = load_images(image_directory)

# Normalize the pixel values to the range [-1, 1]
trainX = (dataset.astype(np.float32) - 127.5) / 127.5
trainY = np.array([[1.0] for _ in trainX])
# Print the shape of the dataset
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)


# Define the generator model
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 8 * 8  # 8x8 feature maps
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (8,8), activation='tanh', padding='same'))
    return model

# Define the discriminator model
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    # normal initialization for stability
    init = RandomNormal(stddev=0.02)
    # downsample to 16x16
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 8x8
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Define the GAN model
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Define the size of the latent space
latent_dim = 100

# Create the generator
generator = define_generator(latent_dim)

# Create the discriminator
discriminator = define_discriminator()

# Create the GAN
gan_model = define_gan(generator, discriminator)

# Train the GAN
def train_gan(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(len(dataset) / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
             summarize_performance(i, generator, discriminator, dataset, latent_dim)

# Generate real samples
def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(f'/Python/470/eclipse_gan/models/{filename}')

# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(f'/Python/470/eclipse_gan/pictures/{filename}')
	plt.close()

# Train the GAN model
train_gan(generator, discriminator, gan_model, trainX, latent_dim)
