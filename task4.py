import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to load CIFAR-10 dataset
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = load_cifar10()

# Load gun images
gun_files = [
    "/Users/giorgia/Downloads/gun1.png",
    "/Users/giorgia/Downloads/gun2.png",
    "/Users/giorgia/Downloads/gun3.png",
    "/Users/giorgia/Downloads/gun4.png",
    "/Users/giorgia/Downloads/gun5.png",
    "/Users/giorgia/Downloads/gun6.png",
    "/Users/giorgia/Downloads/gun7.png",
    "/Users/giorgia/Downloads/gun8.png"
]

gun_images = []
for file in gun_files:
    img = image.load_img(file, target_size=(32, 32))
    img = image.img_to_array(img)
    img = img.astype('float32') / 255.0  # Normalize
    gun_images.append(img)

# Convert gun images to numpy array
gun_images_array = np.array(gun_images)

# Create labels for guns (class 1) and cars (class 0)
gun_labels = np.ones((len(gun_images_array),), dtype=int)
car_labels = np.zeros((len(x_train),), dtype=int)

# Concatenate gun images and car images
x_train = np.concatenate([x_train, gun_images_array])
y_train = np.concatenate([car_labels, gun_labels])

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Predict labels for gun images
gun_predictions = model.predict(gun_images_array)

# Display gun images along with predicted labels
for i, (img, pred) in enumerate(zip(gun_images, gun_predictions)):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    if pred >= 0.5:
        plt.title(f"Gun {i+1}: Gun")
    else:
        plt.title(f"Gun {i+1}: Car")
    plt.axis('off')
    plt.show()
