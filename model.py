import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from random import randint
from scipy.stats import bernoulli

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Strictly define NHWC (Tensorflow) [Theano: NCHW]
from keras import backend as K
K.set_image_dim_ordering('tf')


def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle


def augment_image(image, steering_angle, do_shear_prob=0.9):
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image = random_gamma(image)
    return image, steering_angle


images = []
measurements = []

####
# Reading in 2 laps, counter-clockwise (forward)
####
lines = []
#with open('./Simulation_Data/training1_3img/driving_log.training1_3img.csv') as csvfile:
with open('./Simulation_Data/run1_10Hz_forward_2laps/driving_log.run1_10Hz_forward_2laps.korr.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

for line in lines:
	### Center camera
	#
	source_path = line[0]
	image = plt.imread(source_path)
	measurement = float(line[3])
	# Random augmentation
	image, measurement = augment_image(image, measurement)
	images.append(image)
	measurements.append(measurement)
	### Left Camera
	#
	# Steering correction, determined by manual tests
	steering_corr = 0.015
	source_path = line[1]
	image = plt.imread(source_path)
	measurement = float(line[3])
	measurement += steering_corr
	# Random augmentation
	image, measurement = augment_image(image, measurement)
	images.append(image)
	measurements.append(measurement)
	### Right Camera
	#
	# Steering correction, determined by manual tests
	steering_corr = 0.01
	source_path = line[2]
	image = plt.imread(source_path)
	measurement = float(line[3])
	measurement += steering_corr
	# Random augmentation
	image, measurement = augment_image(image, measurement)
	images.append(image)
	measurements.append(measurement)


####
# Reading in 2 laps, clockwise (rear)
####
if 1:
	lines = []
	#with open('./Simulation_Data/training1_3img/driving_log.training1_3img.csv') as csvfile:
	with open('./Simulation_Data/run2_10Hz_rear_2laps/driving_log.run2_10Hz_rear_2laps.korr.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		### Center Camera
		#
		source_path = line[0]
		image = plt.imread(source_path)
		measurement = float(line[3])
		# Random augmentation
		image, measurement = augment_image(image, measurement)
		images.append(image)
		measurements.append(measurement)
		### Left Camera
		#
		source_path = line[1]
		image = plt.imread(source_path)
		measurement = float(line[3])
		# Random augmentation
		image, measurement = augment_image(image, measurement)
		images.append(image)
		measurements.append(measurement)
		### Right Camera
		#
		source_path = line[2]
		image = plt.imread(source_path)
		measurement = float(line[3])
		# Random augmentation
		image, measurement = augment_image(image, measurement)
		images.append(image)
		measurements.append(measurement)


####
# Reading in extra recordings of S-curve (after the bridge) driving (counter-clockwise)
####
if 1:
	lines = []
	#with open('./Simulation_Data/training1_3img/driving_log.training1_3img.csv') as csvfile:
	with open('./Simulation_Data/run5_10Hz_forward_sline/driving_log.training_curve_corr.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		### Center Camera
		#
		source_path = line[0]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Left Camera
		#
		source_path = line[1]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Right Camera
		#
		source_path = line[2]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)

		
####
# Reading in extra recordings of returning from left and right boundary to center (counter-clockwise)
####
if 1:
	lines = []
	#with open('./Simulation_Data/training1_3img/driving_log.training1_3img.csv') as csvfile:
	with open('./Simulation_Data/run4_10Hz_forward_curvydrv/driving_log.curvy_drv.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		### Center Camera
		#
		source_path = line[0]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Left Camera
		#
		source_path = line[1]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Right Camera
		#
		source_path = line[2]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)

		
####
# Reading in extra recordings of 1 lap, track 2
####
if 1:
	lines = []
	#with open('./Simulation_Data/training1_3img/driving_log.training1_3img.csv') as csvfile:
	with open('./Simulation_Data/run6_10Hz_2ndtrack/driving_log.2ndtrack.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		### Center Camera
		#
		source_path = line[0]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Left Camera
		#
		source_path = line[1]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
		### Right Camera
		#
		source_path = line[2]
		image = plt.imread(source_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

#### NVidia Model
model = Sequential()
# Cropping: (160,320,3) -> (80,320,3)
crop = model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape = (160,320,3), data_format='channels_last'))
# Normalizing
#model.add(Lambda (lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
norm = model.add(Lambda (lambda x: (x / 255.0) - 0.5))
# Layer 1
# Conv2D: 80x320x3 -5x5-> 80x320x24
l1_conv = model.add(Conv2D(24,(5,5), activation='relu', padding='same'))
# Pool:   80x320x24 -2x2-> 40x160x24
l1_pool = model.add(MaxPooling2D((2,2)))
# Layer 2
# Conv2D: 40x160x24 -5x5-> 40x160x36
l2_conv = model.add(Conv2D(36,(5,5), activation='relu', padding='same'))
# Pool: 40x160x36 -2x2-> 20x80x36
l2_pool = model.add(MaxPooling2D((2,2)))
# Layer 3
# Conv2D: 20x80x36 -5x5-> 20x80x48
l3_conv = model.add(Conv2D(48,(5,5), activation='relu', padding='same'))
# Pool: 20x80x48 -2x2-> 10x40x48
l3_pool = model.add(MaxPooling2D((2,2)))
# Layer 4
# Conv2D: 10x40x48 -3x3-> 8x38x64
l4_conv = model.add(Conv2D(64,(3,3), activation='relu', padding='valid'))
# Pool: 8x38x64 -2x2-> 4x19x64
l4_pool = model.add(MaxPooling2D((2,2)))
# Layer 5
# Conv2D: 4x19x64 -3x3-> 2x17x64 
l5_conv = model.add(Conv2D(64,(3,3), activation='relu', padding='valid'))
# Pool:  2x17x64 -3x3-> 1x8x64
l5_pool = model.add(MaxPooling2D((2,2)))
# 1x8x64 -> 512
l6_flat = model.add(Flatten())
# 512 -> 512
#model.add(Dense(1164))
l6_full = model.add(Dense(512))
#model.add(Dropout(0.8))
# 512 -> 100
l7_full = model.add(Dense(100))
#model.add(Dropout(0.8))
# 100 -> 50
l8_full = model.add(Dense(50))
#model.add(Dropout(0.8))
# 50 -> 10
l9_full = model.add(Dense(10))
#model.add(Dropout(0.8))
#  10 -> 1
model.add(Dense(1))


#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(lr=0.0005))
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=8)

model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


