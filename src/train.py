import tensorflow as tf

import os
import time

def train(model, dataset, epochs=10):
    """
        train a model using the dataset
    """
    #define the loss function
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    #train the model
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    
    return history


    
