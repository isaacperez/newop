import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import capa, capa2

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)
"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
"""

"""
model = models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))
model.add(capa.Factor(num_factors=32, kernel_size=3))
model.add(layers.Conv2D(32, (1, 1), activation='relu', use_bias=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=0.5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(capa.Factor(num_factors=32, kernel_size=3))
model.add(layers.Conv2D(32, (1, 1), activation='relu', use_bias=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=0.5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(capa.Factor(num_factors=32, kernel_size=3))
model.add(layers.Conv2D(32, (1, 1), activation='relu', use_bias=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=0.5)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
"""
model = models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
model.add(capa.Factor(num_factors=8, kernel_size=3))
model.add(layers.AveragePooling2D((3, 3)))
model.add(capa.Factor(num_factors=12, kernel_size=3))
model.add(layers.AveragePooling2D((3, 3)))
model.add(capa.Factor(num_factors=16, kernel_size=3))
model.add(layers.GlobalAveragePooling2D())
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=150, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)