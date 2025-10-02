print("Hello!")
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist


# 1. Load your saved model
model = keras.models.load_model("models/mnist_model.h5")

# 2. Load MNIST test data (same as before)
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize test data the same way you did during training
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)  # add channel dimension

# 3. Pick a random test digit
index = np.random.randint(0, len(x_test))
test_image = x_test[index]
true_label = y_test[index]

# 4. Show the image
plt.imshow(test_image.squeeze(), cmap="gray")
plt.title(f"True Label: {true_label}")
plt.savefig("digit.png")
print("Saved digit image as digit.png")


# 5. Make prediction
prediction = model.predict(np.expand_dims(test_image, axis=0))  # add batch dimension
predicted_label = np.argmax(prediction)

print(f"Model prediction: {predicted_label}")
print(f"Actual label: {true_label}")
