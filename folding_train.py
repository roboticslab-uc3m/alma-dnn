import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics, utils

from os import path
from PIL import Image

IMAGE_SIZE = (300, 300, 3)
IMAGE_DIR = "dataset"
BATCH_SIZE = 16
BATCHES = 2 # NUM IMAGES = BATCH_SIZE * BATCHES

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

# Train the model
for idx_outer in range(2):
    print("* begin",idx_outer)
    list_of_arrays = []
    for idx_inner in range(BATCH_SIZE):
        image_name = path.join(IMAGE_DIR,"image"+str(idx_outer*BATCH_SIZE+idx_inner)+".png")
        print("** Loading", image_name)
        array = utils.img_to_array(Image.open(image_name))
        list_of_arrays.append(array)
    X_train = tf.stack(list_of_arrays)
    y_train = np.random.rand(BATCH_SIZE, 4)  # Example target values
    model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE)
    print("* end",idx_outer)

# Example prediction
# X_test should be your test images
X_test = np.random.rand(10, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])  # Example test data
predictions = model.predict(X_test)
print(predictions)
