import tensorflow as tf


class FactorMask(tf.keras.layers.Layer):
    def __init__(self, num_factors, kernel_size):
        super(FactorMask, self).__init__()
        self.num_factors = num_factors
        self.kernel_size = kernel_size
        self.num_elements = self.kernel_size * self.kernel_size
        self.pad_value = int(tf.floor(kernel_size / 2))
        self.paddings = tf.constant([[0, 0], 
                                    [self.pad_value, self.pad_value], 
                                    [self.pad_value, self.pad_value],
                                    [0, 0]])

        self.kernel_initializer = tf.keras.initializers.RandomUniform(minval=0.5, maxval=0.55)
        self.mask_initializer = tf.keras.initializers.RandomUniform(minval=0.75, maxval=0.8)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                    shape=[self.num_elements, 
                                            1, 1, 1,
                                            int(input_shape[-1])],
                                    initializer=self.kernel_initializer)
        self.mask = self.add_weight("mask",
                                    shape=[self.num_elements, 
                                            1, 1, 1,
                                            int(input_shape[-1])],
                                    initializer=self.mask_initializer)

    def call(self, input):

        # Add padding
        padded_input = tf.pad(input, self.paddings)
        batch_size = tf.shape(input)[0]
        height = tf.shape(input)[1]
        width = tf.shape(input)[2]
        
        # Shift the image
        shifted_input = tf.expand_dims(input, axis=0)
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                if y != self.pad_value or x != self.pad_value:
                    shifted_input = tf.concat([shifted_input, 
                        tf.expand_dims(padded_input[:, y:y + height, x:x + width, :], axis=0)], axis=0)
                
        # Transform the kernel to the input size
        current_kernel = tf.tile(self.kernel, multiples=[1, batch_size, height, width, 1])
        current_mask = tf.tile(tf.abs(self.mask) / (tf.reduce_max(tf.abs(self.mask)) + 0.000001), multiples=[1, batch_size, height, width, 1])

        # Do the operation
        shifted_input_masked = shifted_input * current_mask

        abs_diff = tf.abs(shifted_input_masked - current_kernel)
        norm_1 = tf.reduce_sum(abs_diff, axis=0)
        mean_diff = tf.reduce_mean(shifted_input_masked - current_kernel, axis=0)

        variation_diff = tf.reduce_sum(tf.abs(abs_diff - mean_diff), axis=0)
        result = (1 - (variation_diff / self.num_elements)) * (1 - (norm_1 / self.num_elements))

        return result

"""
layer = Factor(num_factors=3, kernel_size=3)
output = layer(tf.zeros([2, 7, 7, 4]))
print(output.shape)
print(layer.trainable_variables[0][:, 0, 0, 0, 0])
import matplotlib.pyplot as plt 

for i in range(output.shape[0]):

    plt.figure(i)
    plt.imshow(output[i, :, :, 0])

plt.show()
"""
"""
print([var.name for var in layer.trainable_variables])
print([var.shape for var in layer.trainable_variables])
print([dir(var) for var in layer.trainable_variables])
"""
"""
I = I(:);
T = T(:);

numElementos = double(length(I));
norma1 = sum(abs(I-T));

mediaDiferencias = mean(I-T);
variacionDeDiferencias = sum(abs((I-T) - mediaDiferencias));

valor = (1-(variacionDeDiferencias/numElementos)).*(1-(norma1/numElementos));
"""