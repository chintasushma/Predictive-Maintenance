import tensorflow as tf
import numpy as np
from Evaluation import evaluation

def Model_DTCN(train_data, train_target, test_data, test_target):
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=1),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(train_data, train_target, epochs=10, batch_size=16)

    # Predict using the trained model
    predictions = model.predict(test_data)
    return predictions

def Model__DTCN(train_data, train_target, test_data, test_target):
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=1),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(train_data, train_target, epochs=10, batch_size=16)

    # Predict using the trained model
    predictions = model.predict(test_data)
    Eval = evaluation(predictions.astype('int'), test_target)
    return Eval

