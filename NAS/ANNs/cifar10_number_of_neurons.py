import tensorflow as tf
import numpy as np


def cifar_normalization(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

def prepare_cifar10_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Normalization
    # Reminder: input_shape = (32, 32, 3)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    x_test = x_test / 255.0

    # Label Encoding
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    return x_train, y_train, x_test, y_test


def get_model(
    first_layer_number_of_neurons, 
    second_layer_number_of_neurons, 
    x_train, 
    y_train,
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    batch_size=32,
    epochs=50
):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(first_layer_number_of_neurons, activation='relu'),
            tf.keras.layers.Dense(second_layer_number_of_neurons, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )

    return history
