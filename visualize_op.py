import cv2
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import quantize_down_and_shrink_range_eager_fallback
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
IMG_PATH = "/home/isaac/workspace/factorizacionCNN/correlacionNueva/input2.png"
TEMPLATE_PATH = "/home/isaac/workspace/factorizacionCNN/correlacionNueva/template2.png"


# -----------------------------------------------------------------------------
# CODE
# -----------------------------------------------------------------------------
def meanDiff(img, template):
    """Both img and template has three dimension"""
    return tf.reduce_mean(tf.abs(img - template), axis=-1)

def meanSquareError(img, template):
    """Both img and template has three dimension"""
    return tf.reduce_mean(tf.pow(img - template, 2), axis=-1)

def crossCorrelationNormed(img, template):
    """Both img and template has one dimension"""
    norm = tf.sqrt(tf.reduce_sum(tf.pow(img, 2), axis=-1) * tf.reduce_sum(tf.pow(template, 2), axis=-1))
    return tf.reduce_sum(img * template, axis=-1) / norm

def calculateMeanSquareError(img, template):

    # Get the original image size
    height_img, width_img, _ = img.shape

    # Calculate the padding for each spatial dimension
    height_template, width_template, _ = template.shape

    height_pad = int(tf.floor(height_template / 2))
    width_pad = int(tf.floor(width_template / 2))

    # Apply the padding (only to the spatial dimensions, channels doesn't need padding)
    img_padded = tf.pad(img, [[height_pad, height_pad], [width_pad, width_pad], [0, 0]]) 

    # Shift the img to simulate the movement of the template
    shifted_img = img
    shifted_template = template[height_pad, width_pad, :]
    for row in range(height_template):
        for col in range(width_template):
            if row != height_pad or col != width_pad:
                shifted_img = tf.concat([shifted_img, img_padded[row:row + height_img, col:col + width_img, :]], axis=-1)
                shifted_template = tf.concat([shifted_template, template[row, col, :]], axis=-1)

    # Add the spatial dimensions height and width    
    shifted_template = tf.expand_dims(tf.expand_dims(shifted_template, 0), 0)

    # Replicate the 1x1x(3*n) template to has the spatial dimension of the image
    shifted_template = tf.tile(shifted_template, shifted_img.shape[:-1] + [1]) 
    
    # Now we can calculate the value
    mean_square_error = meanSquareError(shifted_img, shifted_template)

    return mean_square_error


def calculateCrossCorrelationNormed(img, template):

    # Get the original image size
    height_img, width_img, _ = img.shape

    # Calculate the padding for each spatial dimension
    height_template, width_template, _ = template.shape

    height_pad = int(tf.floor(height_template / 2))
    width_pad = int(tf.floor(width_template / 2))

    # Apply the padding (only to the spatial dimensions, channels doesn't need padding)
    img_padded = tf.pad(img, [[height_pad, height_pad], [width_pad, width_pad], [0, 0]]) 

    # Shift the img to simulate the movement of the template
    shifted_img = img
    shifted_template = template[height_pad, width_pad, :]
    for row in range(height_template):
        for col in range(width_template):
            if row != height_pad or col != width_pad:
                shifted_img = tf.concat([shifted_img, img_padded[row:row + height_img, col:col + width_img, :]], axis=-1)
                shifted_template = tf.concat([shifted_template, template[row, col, :]], axis=-1)

    # Add the spatial dimensions height and width    
    shifted_template = tf.expand_dims(tf.expand_dims(shifted_template, 0), 0)

    # Replicate the 1x1x(3*n) template to has the spatial dimension of the image
    shifted_template = tf.tile(shifted_template, shifted_img.shape[:-1] + [1]) 
    
    # Now we can calculate the value
    cross_correlation_normed = crossCorrelationNormed(shifted_img, shifted_template)

    return cross_correlation_normed


def calculateFactorization(img, template):

    # Get the original image size
    height_img, width_img, _ = img.shape

    # Calculate the padding for each spatial dimension
    height_template, width_template, _ = template.shape

    height_pad = int(tf.floor(height_template / 2))
    width_pad = int(tf.floor(width_template / 2))

    # Apply the padding (only to the spatial dimensions, channels doesn't need padding)
    img_padded = tf.pad(img, [[height_pad, height_pad], [width_pad, width_pad], [0, 0]]) 

    # Shift the img to simulate the movement of the template
    shifted_img = img
    shifted_template = template[height_pad, width_pad, :]
    for row in range(height_template):
        for col in range(width_template):
            if row != height_pad or col != width_pad:
                shifted_img = tf.concat([shifted_img, img_padded[row:row + height_img, col:col + width_img, :]], axis=-1)
                shifted_template = tf.concat([shifted_template, template[row, col, :]], axis=-1)

    # Add the spatial dimensions height and width    
    shifted_template = tf.expand_dims(tf.expand_dims(shifted_template, 0), 0)

    # Replicate the 1x1x(3*n) template to has the spatial dimension of the image
    shifted_template = tf.tile(shifted_template, shifted_img.shape[:-1] + [1]) 
    
    # Now we can calculate the value
    cross_correlation_normed = crossCorrelationNormed(shifted_img, shifted_template)
    mean_square_error = meanSquareError(shifted_img, shifted_template)

    factorization = (1 - mean_square_error) * cross_correlation_normed

    # The original version
    """
    norma1 = tf.reduce_mean(tf.abs(shifted_img - shifted_template), axis=-1)

    mediaDiferencias = tf.reduce_mean(shifted_img - shifted_template)
    variacionDeDiferencias = tf.reduce_mean(tf.abs((shifted_img - shifted_template) - mediaDiferencias))

    original_factorization = (1 - variacionDeDiferencias) * (1 - norma1)
    """
    cross_correlation_normed = crossCorrelationNormed(shifted_img, shifted_template)
    mean_diff = meanDiff(shifted_img, shifted_template)

    original_factorization = (1 - mean_diff) * cross_correlation_normed

    return factorization, original_factorization


# Read the input image and template
img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(cv2.imread(TEMPLATE_PATH), cv2.COLOR_BGR2RGB)

# Transform to float and move to the range [0, 1]
img = img.astype(np.float32) / 255.0
template = template.astype(np.float32) / 255.0

# First, we are going to calculate the mean square error
mean_square_error = calculateMeanSquareError(img, template)

# and the cross corrlation normed
cross_correlation_normed = calculateCrossCorrelationNormed(img, template)

# So we can now calculate the factorization
factorization, original_factorization = calculateFactorization(img, template)

# Show the results
plt.figure(1, figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Input image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Template")
plt.imshow(template)

plt.figure(3)
plt.suptitle("Mean Square Error")
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(mean_square_error)
plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(mean_square_error > 0.5)
plt.colorbar()

plt.figure(4)
plt.axis("off")
plt.suptitle("Cross Correlation Normed")
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(cross_correlation_normed)
plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(cross_correlation_normed > 0.75)
plt.colorbar()

plt.figure(5)
plt.axis("off")
plt.suptitle("Factorization")
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(factorization)
plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(factorization > 0.75)
plt.colorbar()

plt.figure(6)
plt.axis("off")
plt.title("Factorization original")
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(original_factorization)
plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(original_factorization > 0.75)
plt.colorbar()

for (tensor, name) in [(mean_square_error, "Mean Square Error"), (cross_correlation_normed, "Cross Correlation Normed"),
                       (factorization, "Factorization"), (original_factorization, "Factorization original")]:            
    tensor = tensor.numpy()
    max_value = np.max(tensor)
    idx_max = np.unravel_index(tensor.argmax(), tensor.shape)
    print(name, "\n\tMax. value = ", max_value, " - ", np.flip(idx_max), sep="")

plt.show()