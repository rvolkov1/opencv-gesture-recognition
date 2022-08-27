import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import Data
import numpy as np
import os

class GestureModel():
    def __init__(self, model_name=None):
        self.filename = "./person1-gesture-model"
        self.model = None
        self.gesture_names = ["hand", "fist", "one finger", "two fingers", "three fingers"]

        if (os.path.exists(self.filename)):
            self.model = self.load_model()
        if (os.path.exists("../data/landmarks.csv")):
            self.create_model()
            self.train_model()

    def create_model(self):
        inputs = keras.Input(shape=(63,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        x = layers.Dense(64, activation="relu", name="dense_3")(x)
        outputs = layers.Dense(5, activation="softmax", name="predictions")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    def train_model(self):
        data = Data()
        x, y = data.load_data()

        x_train = np.array(x)
        y_train = np.array(y)

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=10,
            epochs=2,
        )
        return history

    def save_model(self):
        self.model.save(self.filename)
    
    def load_model(self):
        keras.models.load_model(self.filename)
        return self.model
    
    def predict(self, landmarks):
        return self.model.predict(np.array(landmarks))







