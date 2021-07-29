import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import baseModel 


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
WIDTH = 28
HEIGHT = 28
CHANNELS = 1

LEARNING_RATE = 0.001 
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
#OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
BATCH_SIZE = int(32)
NUM_EPOCH = int(50)

IDX_TRACING_IMG = 0


# -----------------------------------------------------------------------------
# Code
# -----------------------------------------------------------------------------

# Build the dataset
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

batches_in_the_train_dataset = int(tf.math.floor(train_images.shape[0] / BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

batches_in_the_test_dataset = int(tf.math.floor(test_images.shape[0] / BATCH_SIZE))

# Built the model
current_model = baseModel.BaseModel()
current_model.build((None, HEIGHT, WIDTH, CHANNELS+2))
print(current_model.summary())

model_output = current_model(tf.tile(tf.expand_dims(test_images[IDX_TRACING_IMG, :, :, :], axis=0), multiples=[1,1,1,3]), training=tf.constant(False))
exit()
# Begining training
print("Begining training...")

@tf.function
def train_step(img):

    with tf.GradientTape() as tape:

        # Get the output of the model
        model_output = current_model(img, training=tf.constant(True))


        # Calculate the loss
        max_by_channels = tf.reduce_max(model_output, axis=-1)

        max_by_channels_inverted = 1 - max_by_channels  # We are minimizing no maximizing

        #mean_by_channels_without_max = (tf.reduce_sum(model_output, axis=-1) - max_by_channels) / current_model.num_factors

        #mean_training_loss = tf.reduce_mean(max_by_channels_inverted * mean_by_channels_without_max)
        mean_training_loss = tf.reduce_mean(max_by_channels_inverted)

    # Calculate the gradients
    grads = tape.gradient(mean_training_loss, current_model.trainable_variables)

    # Do the backward pass and adjust weights
    OPTIMIZER.apply_gradients(zip(grads, current_model.trainable_variables))

    return mean_training_loss

# Training loop
fig = plt.figure(0)
subplot = fig.add_subplot(current_model.num_factors + 1, 3, 1)
subplot.set_title("Original image", fontsize=12)
plt.imshow(test_images[IDX_TRACING_IMG, :, :, 0], cmap='gray', vmin=0, vmax=1)
subplot.axis('off')
plt.tight_layout()
plt.draw()
plt.pause(0.001)

best_test_loss = -1
for epoch in range(NUM_EPOCH):

    # Do the validation after the training dataset
    print("[EPOCH " + str(epoch).ljust(6) + " of "
                + str(NUM_EPOCH) + "]")

    it_train = 0
    epoch_mean_training_loss = 0.0
    for img, expected_output in train_dataset:

        # Do the train step
        mean_training_loss = train_step(img)
        
        epoch_mean_training_loss += mean_training_loss.numpy()

        # Show the results of the current iteration
        print("[EPOCH " + str(epoch).ljust(6) + " / "
                    + str(NUM_EPOCH) + "][TRAIN It: "
                    + str(it_train).ljust(6) + " / " + str(batches_in_the_train_dataset)  +"]: "
                    + str(np.round(mean_training_loss.numpy(), 4)).ljust(6), end="\r")

        it_train += 1

    print("[EPOCH " + str(epoch).ljust(6) + " / "
                    + str(NUM_EPOCH) + "] Mean loss train  =", epoch_mean_training_loss / it_train)

    # Show the results of the tracing image
    model_output = current_model(tf.expand_dims(test_images[IDX_TRACING_IMG, :, :, :], axis=0), training=tf.constant(False))

    subplot = fig.add_subplot(current_model.num_factors + 1, 3, 2)
    subplot.set_title("argmax ", fontsize=12)
    plt.imshow(tf.argmax(model_output, axis=-1)[0, :, :], cmap='viridis', vmin=0, vmax=current_model.num_factors-1)
    subplot.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    model_output_one_hot = tf.one_hot(tf.argmax(model_output, axis=-1), depth=current_model.num_factors)

    for i in range(current_model.num_factors):

        subplot = fig.add_subplot(current_model.num_factors + 1, 3, 4 + i * 3)
        subplot.set_title("Factor " + str(i+1), fontsize=12)
        current_factor = current_model.factorLayer.trainable_variables[0].numpy()[:, 0, 0, 0, 0, i].reshape((3,3))
        plt.imshow(np.clip(current_factor, 0.0, 1.0), cmap='gray', vmin=0, vmax=1)
        subplot.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        subplot = fig.add_subplot(current_model.num_factors + 1, 3, 4 + i * 3 + 1)
        subplot.set_title("Factor " + str(i+1) + " output", fontsize=12)
        plt.imshow(model_output[0, :, :, i], cmap='gray', vmin=0, vmax=1)
        subplot.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        subplot = fig.add_subplot(current_model.num_factors + 1, 3, 4 + i * 3 + 2)
        subplot.set_title("Factor " + str(i+1) + " output one-hot", fontsize=12)
        plt.imshow(model_output_one_hot[0, :, :, i], cmap='gray', vmin=0, vmax=1)
        subplot.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)



    # Do the validation
    it_test = 0
    epoch_mean_training_loss = 0.0
    for img, expected_output in test_dataset:

        # Get the model output
        model_output = current_model(img, training=tf.constant(False))

        # Calculate the loss
        max_by_channels = tf.reduce_max(model_output, axis=-1)
        max_by_channels_inverted = 1 - max_by_channels  # We are minimizing no maximizing
        mean_training_loss = tf.reduce_mean(max_by_channels_inverted)
        
        epoch_mean_training_loss += mean_training_loss.numpy()

        print(str(it_test).ljust(6), "of", batches_in_the_test_dataset, end="\r")

        it_test += 1
    last_test_loss = epoch_mean_training_loss / it_test
    print("[EPOCH " + str(epoch).ljust(6) + " / "
                    + str(NUM_EPOCH) + "] Mean loss test =", last_test_loss)

    if best_test_loss == -1 or last_test_loss < best_test_loss:
        best_test_loss = last_test_loss
        current_model.save_weights("./best_test_loss")


plt.show()