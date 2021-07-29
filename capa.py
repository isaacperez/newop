import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')


class Factor(tf.keras.layers.Layer):
    def __init__(self, num_factors, kernel_size):
        super(Factor, self).__init__()
        # The number of output channels to generate a.k.a. num_factors
        self.num_factors = num_factors

        # The spatial size of the pattern (it's the same for width and height)
        self.kernel_size = kernel_size

        # The number of weights that are used for one factor and one input's channel
        self.num_elements_by_factor = self.kernel_size * self.kernel_size

        # The output size is the same as the input
        self.pad_value = int(tf.floor(kernel_size / 2))
        self.paddings = tf.constant([[0, 0], 
                                    [self.pad_value, self.pad_value], 
                                    [self.pad_value, self.pad_value],
                                    [0, 0]])
        
        # The initializer for the kernel
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=0.1)

    def build(self, input_shape):
        # The size of the kernel is the spatial size x the input's channels x the output's channels
        self.kernel = self.add_weight("kernel",
                                    shape=[self.num_elements_by_factor, 
                                            1, 1, 1,
                                            int(input_shape[-1]), self.num_factors],
                                    initializer=self.initializer)

    @tf.function
    def call(self, input):

        # Add padding
        padded_input = tf.pad(input, self.paddings)
        batch_size = tf.shape(input)[0]
        height = tf.shape(input)[1]
        width = tf.shape(input)[2]
        channels = tf.shape(input)[3]
        
        # Shift the image
        shifted_input = input
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                if y != self.pad_value or x != self.pad_value:
                    shifted_input = tf.concat([shifted_input, padded_input[:, y:y + height, x:x + width, :]], axis=-1)
        
        return shifted_input
        # Group each channel shift in order
        """

        # Calculate each factor

        # Transform the kernel to the input size
        kernel_expanded = tf.tile(self.kernel, multiples=[1, batch_size, height, width, 1, 1])

        # Do the operation
        current_kernel = kernel_expanded[:, :, :, :, :, 0]
        abs_diff = tf.abs(shifted_input - current_kernel)
        norm_1 = tf.reduce_mean(abs_diff, axis=0)
        mean_diff = tf.abs(tf.reduce_mean(shifted_input - current_kernel, axis=0))
        
        variation_diff = tf.reduce_mean(tf.abs(abs_diff - mean_diff), axis=0)
        result = (1 - variation_diff) * (1 - norm_1)
        tf.print(tf.shape(abs_diff))
        tf.print(tf.shape(norm_1))
        tf.print(tf.shape(mean_diff))
        tf.print(tf.shape(variation_diff))
        tf.print(tf.shape(result))
        for i in range(1, self.num_factors):
            current_kernel = kernel_expanded[:, :, :, :, :, i]

            # Do the operation
            abs_diff = tf.abs(shifted_input - current_kernel)
            norm_1 = tf.reduce_mean(abs_diff, axis=0)
            mean_diff = tf.abs(tf.reduce_mean(shifted_input - current_kernel, axis=0))

            variation_diff = tf.reduce_mean(tf.abs(abs_diff - mean_diff), axis=0)
            current_result = (1 - variation_diff) * (1 - norm_1)

            result = tf.concat([result, current_result], axis=-1)

        return result
        """

# Code for testing
layer = Factor(num_factors=3, kernel_size=3)
output = layer(tf.zeros([2, 7, 7, 4]))
print(output.shape)
print(layer.trainable_variables[0][:, 0, 0, 0, 0])
"""
import matplotlib.pyplot as plt 

for i in range(output.shape[0]):

    plt.figure(i)
    plt.imshow(output[i, :, :, 0])

plt.show()
"""

print([var.name for var in layer.trainable_variables])
print([var.shape for var in layer.trainable_variables])
print([dir(var) for var in layer.trainable_variables])

"""
I = I(:);
T = T(:);

numElementos = double(length(I));
norma1 = sum(abs(I-T));

mediaDiferencias = mean(I-T);
variacionDeDiferencias = sum(abs((I-T) - mediaDiferencias));

valor = (1-(variacionDeDiferencias/numElementos)).*(1-(norma1/numElementos));
"""