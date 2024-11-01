import tensorflow as tf


def prepare_cifar10_data():
    num_classes = 10

    # Load data
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_train  /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    return x_train, y_train, x_test, y_test


def get_model(
    first_layer_number_of_neurons, 
    second_layer_number_of_neurons, 
    x_train, 
    y_train,
    optimizer='sgd',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'],
    batch_size=32,
    epochs=10
):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(first_layer_number_of_neurons, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(second_layer_number_of_neurons, activation='relu'),
            tf.keras.layers.Dropout(0.2),
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
