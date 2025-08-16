import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 6
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
# NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    res = load_data(sys.argv[1])
    if res is None:
        exit("Data loading went wrong")

    images, labels = res

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, random_state=42
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    try:
        model.fit(x_train, y_train, epochs=EPOCHS)
    except KeyboardInterrupt:
        print("Interrupt")
        exit(0)

    # Evaluate neural network performance
    model.evaluate(x_train, y_train, verbose=2)
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir: str) -> tuple[list[np.ndarray], list[int]] | None:
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        dir = os.path.join(data_dir, str(category))

        if not os.path.isdir(dir):
            return None

        for name in os.listdir(dir):
            if not name.lower().endswith(".ppm"):
                print("continue")
                continue

            path = os.path.join(dir, name)
            if not os.path.isfile(path):
                return None

            try:
                img = cv2.imread(path)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
                assert resized.shape == (30, 30, 3)

                images.append(resized)
                labels.append(category)
            except Exception as e:
                print(f"Could not read {name}: {e}")

    assert len(images) == len(labels)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    main()
