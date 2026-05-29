import tensorflow as tf
from tensorflow.keras import layers, models

def create_brain():
    model = models.Sequential()
    
    # Input Shape: (4 timesteps, 5 height, 5 width, 1 channel)
    model.add(layers.InputLayer(shape=(4, 5, 5, 1)))
    
    # Process each frame through the CNN 
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.Flatten()))
    
    # The LSTM reads the sequence of the 4 processed frames to detect motion/wiggling
    model.add(layers.LSTM(32, activation='tanh'))
    
    # Output layer for the 4 joystick actions
    model.add(layers.Dense(4, activation='linear'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    return model