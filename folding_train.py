import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics, utils

from os import path
from PIL import Image

DATASET_DIR = "dataset"

IMAGE_SIZE = (300, 300, 3)
BATCH_SIZE = 10 # 16
NUM_TRAIN_BATCHES = 2 # NUM TRAIN_IMAGES = BATCH_SIZE * NUM_TRAIN_BATCHES
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

def load_image_batch(idx_init, batch_size):
    list_of_arrays = []
    for idx_inner in range(batch_size):
        image_name = path.join(DATASET_DIR,"image"+str(idx_init+idx_inner)+".png")
        print("** list_of_arrays["+str(idx_inner)+"]", image_name)
        array = utils.img_to_array(Image.open(image_name))
        list_of_arrays.append(array)
    return tf.stack(list_of_arrays)

# Train the model
for idx_outer in range(NUM_TRAIN_BATCHES):
    print("* begin train batch",idx_outer)
    print("** x_train")
    x_train = load_image_batch(idx_outer*BATCH_SIZE, BATCH_SIZE)
    y_train = labels[idx_outer*BATCH_SIZE:(idx_outer+1)*BATCH_SIZE]
    print("** y_train", y_train)
    model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE)
    print("* end train batch",idx_outer)

# Example prediction
print("* begin test")
print("** x_test")
x_test = load_image_batch(NUM_TRAIN_BATCHES*BATCH_SIZE, NUM_TEST_BATCHES*BATCH_SIZE)
predictions = model.predict(x_test)
print(predictions)
print("* end test")
