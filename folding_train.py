import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics, utils

from os import path
from PIL import Image

DATASET_DIR = "dataset"

IMAGE_SIZE = (100, 100, 1) # (100, 100, 3)
BATCH_SIZE = 10 # 16
NUM_TRAIN_BATCHES = 1000 # NUM TRAIN_IMAGES = BATCH_SIZE * NUM_TRAIN_BATCHES
NUM_TEST_BATCHES = 1 # NUM TEST_IMAGES = BATCH_SIZE * NUM_TEST_BATCHES

labels = np.loadtxt(path.join(DATASET_DIR,"labels.txt"), delimiter=',', usecols=range(1,5))
#print("labels",labels)

def define_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(IMAGE_SIZE[0], activation='relu'),
        layers.Dense(4)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.MeanSquaredError(), metrics.AUC()])
    return model
model = define_model()

def load_images(idx_init, num_images):
    list_of_arrays = []
    for idx_inner in range(num_images):
        image_name = path.join(DATASET_DIR,"image"+str(idx_init+idx_inner)+".png")
        #j#print("** list_of_arrays["+str(idx_inner)+"]", image_name)
        array = utils.img_to_array(Image.open(image_name))
        array = array[:,:,0] # (100, 100, 3) -> (100, 100)
        #j#getMaxVal#max_val = tf.reduce_max(array, keepdims=True)
        array /= 200.0 # normalize
        list_of_arrays.append(array)
    return tf.stack(list_of_arrays)

# Train the model
"""
for idx_outer in range(NUM_TRAIN_BATCHES):
    print("* begin train batch",idx_outer)
    #j#print("** x_train")
    x_train = load_images(idx_outer*BATCH_SIZE, BATCH_SIZE)
    y_train = labels[idx_outer*BATCH_SIZE:(idx_outer+1)*BATCH_SIZE]
    #j#print("** y_train", y_train)
    loss = model.train_on_batch(x_train, y_train)
    print("** train loss",loss)
    print("* end train batch",idx_outer)
"""

x_train = load_images(0, NUM_TRAIN_BATCHES*BATCH_SIZE)
y_train = labels[0:NUM_TRAIN_BATCHES*BATCH_SIZE]
model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE)

# Test the model
print("* begin test")
x_test = load_images(NUM_TRAIN_BATCHES*BATCH_SIZE, NUM_TEST_BATCHES*BATCH_SIZE)
y_test = labels[NUM_TRAIN_BATCHES*BATCH_SIZE:(NUM_TEST_BATCHES+NUM_TRAIN_BATCHES)*BATCH_SIZE]
loss = model.test_on_batch(x_test, y_test)
print("** test loss", loss)
print("* end test")

# Example prediction
print("* begin prediciton")
print("** using same x_test")
predictions = model.predict(x_test)
print(predictions)
print("* end prediciton")
