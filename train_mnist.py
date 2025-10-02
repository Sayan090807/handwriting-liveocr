import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def load_data():
    # Loads MNIST from Keras

    # X_train -> train images, y_train -> train labels, x_test -> test images, y_test -> test labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess(x_train, x_test):
    # Convert pixel values from 0-255 to 0-1
    # Add a channel dimension so images are (28, 28, 1) instead of (28, 28)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (grayscale = 1 channel) channels = number of color layers

    # CNNs expect channel dimensions. In this case since we're using grayscale, we only need 1 instead of 3
    x_train = np.expand_dims(x_train, axis = -1) # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis = -1) # (N, 28, 28, 1)
    
    return x_train, x_test

def build_model():
    # A small CNN
    # Conv + Pool layers learn visual features (edges, curves)
    # Flatten + Dense layers turn features into a digit prediction (0-9)

    model = models.Sequential([
        # Convolutional Layer (detects patterns)
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        # Downsampling
        layers.MaxPooling2D((2, 2)),
        # Another convolutional layer with 64 filters (takes max values)
        layers.Conv2D(64, (3, 3), activation="relu"),
        # Another pooling layer, halves the size again
        layers.MaxPooling2D((2, 2)),
        # Takes @D feature maps and unrolls them into 1D vectors
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")  # 10 classes for digits 0-9
    ])

    model.compile(
        # uses Adam algorithm to adjust the network's weights efficiently
        optimizer="adam",
        # Loss measures how wrong the model is
        loss="sparse_categorical_crossentropy",
        # Tells Keras to report accuracy while training
        metrics=["accuracy"]
    )
    return model

def main(): 
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Preprocess images
    x_train, x_test = preprocess(x_train, x_test)

    # Build CNN
    model = build_model()
    model.summary() # Prints a table of layers and params

    # Train the model
    # - epochs: how many passes over the training data
    # - validation_data: lets us see accuracy on test set while training
    
    history = model.fit(
        x_train, y_train, 
        epochs = 5,
        batch_size = 64,
        validation_data  = (x_test, y_test)
    )

    # Evaluate final accuracy using test set

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    # Save the model for webcam

    os.makedirs("models", exist_ok = True)
    save_path = 'models/mnist_model.h5'
    model.save(save_path)
    print(f"Saved model to {save_path}")

    # Predict one sample
    idx = 0 # try the first test image
    sample = x_test[idx:idx + 1] # shape (1, 28, 28, 1)
    pred_probs = model.predict(sample, verbose = 0)
    pred_label = int(np.argmax(pred_probs, axis = 1)[0])
    true_label = int(y_test[idx])
    print(f"Example Prediction -> predicted: {pred_label}, true: {true_label}")

if __name__ == "__main__":
    main()


